use crate::bsdfs::reflect_vector;
use crate::emitter::*;
use crate::integrators::gradient::*;
use crate::integrators::*;
use cgmath::*;

pub struct IntegratorGradientPath {
    pub max_depth: Option<u32>,
    pub min_depth: Option<u32>,
    pub recons: Box<dyn PoissonReconstruction + Sync>,
}

struct RayStateData<'a> {
    pub pdf: f64,
    pub ray: Ray,
    pub its: Intersection<'a>,
    pub throughput: Color,
}

enum RayState<'a> {
    NotConnected(RayStateData<'a>),
    RecentlyConnected(RayStateData<'a>),
    Connected(RayStateData<'a>),
    // FIXME: Do we need to store all the data?
    Dead,
}

impl<'a> RayState<'a> {
    pub fn check_normal(self) -> RayState<'a> {
        // FIXME: Change how this works .... to avoid duplicated code
        match self {
            RayState::NotConnected(e) => {
                if e.its.cos_theta() <= 0.0 {
                    RayState::Dead
                } else {
                    RayState::NotConnected(e)
                }
            }
            RayState::RecentlyConnected(e) => {
                if e.its.n_s.dot(e.ray.d) > 0.0 {
                    RayState::Dead
                } else {
                    RayState::RecentlyConnected(e)
                }
            }
            RayState::Connected(e) => {
                // FIXME: Maybe not a good idea...
                // FIXME: As the shift path may be not used anymore
                assert!(e.its.n_s.dot(e.ray.d) <= 0.0);
                RayState::Connected(e)
            }
            RayState::Dead => RayState::Dead,
        }
    }

    pub fn apply_russian_roulette(&mut self, rr_prob: f32) {
        match self {
            RayState::Dead => {}
            RayState::NotConnected(ref mut e)
            | RayState::Connected(ref mut e)
            | RayState::RecentlyConnected(ref mut e) => {
                e.throughput /= rr_prob;
            }
        }
    }

    pub fn new(
        (x, y): (f32, f32),
        off: Point2<i32>,
        accel: &'a dyn Acceleration,
        scene: &'a Scene,
    ) -> RayState<'a> {
        let pix = Point2::new(x + off.x as f32, y + off.y as f32);
        if pix.x < 0.0
            || pix.x > (scene.camera.size().x as f32)
            || pix.y < 0.0
            || pix.y > (scene.camera.size().y as f32)
        {
            return RayState::Dead;
        }

        let ray = scene.camera.generate(pix);
        let its = match accel.trace(&ray) {
            Some(x) => x,
            None => return RayState::Dead,
        };

        RayState::NotConnected(RayStateData {
            pdf: 1.0,
            ray,
            its,
            throughput: Color::one(),
        })
    }
}

impl Integrator for IntegratorGradientPath {}
impl IntegratorGradient for IntegratorGradientPath {
    fn reconstruct(&self) -> &Box<dyn PoissonReconstruction + Sync> {
        &self.recons
    }

    fn compute_gradients(&mut self, accel: &dyn Acceleration, scene: &Scene) -> BufferCollection {
        let (nb_buffers, buffernames, mut image_blocks, ids) =
            generate_img_blocks_gradient(scene, &self.recons);

        let progress_bar = Mutex::new(ProgressBar::new(image_blocks.len() as u64));
        let pool = generate_pool(scene);
        pool.install(|| {
            image_blocks.par_iter_mut().for_each(|(info, im_block)| {
                let emitters = scene.emitters_sampler();
                let mut sampler = independent::IndependentSampler::default();
                for ix in info.x_pos_off..im_block.size.x - info.x_size_off {
                    for iy in info.y_pos_off..im_block.size.y - info.y_size_off {
                        for n in 0..scene.nb_samples {
                            let c = self.compute_pixel(
                                (ix + im_block.pos.x, iy + im_block.pos.y),
                                accel,
                                scene,
                                &emitters,
                                &mut sampler,
                            );
                            // Accumulate the values inside the buffer
                            let pos = Point2::new(ix, iy);
                            let offset_buffers = (n % nb_buffers) * 3; // 3 buffers are in multiple version
                            im_block.accumulate(
                                pos,
                                c.main,
                                &buffernames[ids.primal + offset_buffers],
                            );
                            im_block.accumulate(
                                pos,
                                c.very_direct,
                                &buffernames[ids.very_direct].to_owned(),
                            );
                            for i in 0..4 {
                                // primal reuse
                                let off = GRADIENT_ORDER[i];
                                let pos_off = Point2::new(ix as i32 + off.x, iy as i32 + off.y);
                                im_block.accumulate_safe(
                                    pos_off,
                                    c.radiances[i],
                                    &buffernames[ids.primal + offset_buffers],
                                );
                                // gradient
                                match GRADIENT_DIRECTION[i] {
                                    GradientDirection::X(v) => match v {
                                        1 => im_block.accumulate(
                                            pos,
                                            c.gradients[i],
                                            &buffernames[ids.gradient_x + offset_buffers],
                                        ),
                                        -1 => im_block.accumulate_safe(
                                            pos_off,
                                            c.gradients[i] * -1.0,
                                            &buffernames[ids.gradient_x + offset_buffers],
                                        ),
                                        _ => panic!("wrong displacement X"), // FIXME: Fix the enum
                                    },
                                    GradientDirection::Y(v) => match v {
                                        1 => im_block.accumulate(
                                            pos,
                                            c.gradients[i],
                                            &buffernames[ids.gradient_y + offset_buffers],
                                        ),
                                        -1 => im_block.accumulate_safe(
                                            pos_off,
                                            c.gradients[i] * -1.0,
                                            &buffernames[ids.gradient_y + offset_buffers],
                                        ),
                                        _ => panic!("wrong displacement Y"),
                                    },
                                }
                            }
                        }
                    }
                }
                im_block.scale(1.0 / (scene.nb_samples as f32));
                // Renormalize correctly the buffer informations
                for i in 0..nb_buffers {
                    let offset_buffers = i * 3; // 3 buffer that have multiple entries
                                                // 4 strategies as reuse primal
                    im_block.scale_buffer(
                        0.25 * nb_buffers as f32,
                        &buffernames[ids.primal + offset_buffers],
                    );
                    im_block.scale_buffer(
                        nb_buffers as f32,
                        &buffernames[ids.gradient_x + offset_buffers],
                    );
                    im_block.scale_buffer(
                        nb_buffers as f32,
                        &buffernames[ids.gradient_y + offset_buffers],
                    );
                }

                {
                    progress_bar.lock().unwrap().inc();
                }
            });
        });

        // Fill the image & do the reconstruct
        let mut image =
            BufferCollection::new(Point2::new(0, 0), *scene.camera.size(), &buffernames);
        for (_, im_block) in &image_blocks {
            image.accumulate_bitmap(im_block);
        }
        image
    }
}

impl IntegratorGradientPath {
    fn compute_pixel(
        &self,
        (ix, iy): (u32, u32),
        accel: &dyn Acceleration,
        scene: &Scene,
        emitters: &EmitterSampler,
        sampler: &mut dyn Sampler,
    ) -> ColorGradient {
        let mut l_i = ColorGradient::default();
        let pix = (ix as f32 + sampler.next(), iy as f32 + sampler.next());
        let mut main = match RayState::new(pix, Point2::new(0, 0), accel, scene) {
            RayState::NotConnected(x) => x,
            _ => return l_i,
        };
        let mut offsets: Vec<RayState> = {
            GRADIENT_ORDER
                .iter()
                .map(|e| RayState::new(pix, *e, accel, scene))
                .collect()
        };

        // Use the balance heuristic for now
        const MIS_POWER: i32 = 1;

        // For now, just replay the random numbers
        let mut depth: u32 = 1;
        while self.max_depth.is_none() || (depth < self.max_depth.unwrap()) {
            // Check if we go the right orientation
            // -- main path
            if main.its.cos_theta() <= 0.0 {
                return l_i;
            }
            offsets = offsets.into_iter().map(|e| e.check_normal()).collect();

            // Add the emission for the light intersection
            if self.min_depth.map_or(true, |min| depth >= min) && depth == 1 {
                l_i.very_direct += &main.its.mesh.emission; // TODO: Add throughput
            }

            /////////////////////////////////
            // Light sampling
            /////////////////////////////////
            // Explict connect to the light source
            if !main.its.mesh.bsdf.is_smooth() {
                let (r_sel_rand, r_rand, uv_rand) =
                    (sampler.next(), sampler.next(), sampler.next2d());
                let main_light_record =
                    emitters.sample_light(&main.its.p, r_sel_rand, r_rand, uv_rand);
                let main_light_visible = accel.visible(&main.its.p, &main_light_record.p);
                let main_emitter_rad = if main_light_visible {
                    main_light_record.weight
                } else {
                    Color::zero()
                };
                let main_d_out_local = main.its.frame.to_local(main_light_record.d);
                // Evaluate BSDF values and light values
                let main_light_pdf = f64::from(main_light_record.pdf.value());
                let main_bsdf_value = main.its.mesh.bsdf.eval(
                    &main.its.uv,
                    &main.its.wi,
                    &main_d_out_local,
                    Domain::SolidAngle,
                ); // f(...) * cos(...)
                let main_bsdf_pdf = if main_light_visible {
                    f64::from(
                        main.its
                            .mesh
                            .bsdf
                            .pdf(
                                &main.its.uv,
                                &main.its.wi,
                                &main_d_out_local,
                                Domain::SolidAngle,
                            )
                            .value(),
                    )
                } else {
                    0.0
                };

                // Cache PDF / throughput values
                let main_weight_num = main_light_pdf.powi(MIS_POWER);
                let main_weight_dem =
                    main_light_pdf.powi(MIS_POWER) + main_bsdf_pdf.powi(MIS_POWER);
                let main_contrib = main.throughput * main_bsdf_value * main_emitter_rad;
                // Cache geometric informations
                let main_geom_dsquared = (main.its.p - main_light_record.p).magnitude2();
                let main_geom_cos_light = main_light_record.n.dot(main_light_record.d);

                if main_light_record.pdf.value() != 0.0 {
                    // Do the actual shift mapping
                    for (i, offset) in offsets.iter().enumerate() {
                        let (shift_weight_dem, shift_contrib) = match offset {
                            RayState::Dead => {
                                (main_weight_num / (0.0001 + main_weight_dem), Color::zero())
                            }
                            RayState::Connected(ref s) => {
                                // Just reuse all the computation from the base path
                                let shift_weight_dem = (s.pdf / main.pdf).powi(MIS_POWER)
                                    * (main_light_pdf.powi(MIS_POWER)
                                        + main_bsdf_pdf.powi(MIS_POWER));
                                let shift_contrib =
                                    s.throughput * main_bsdf_value * main_emitter_rad;
                                (shift_weight_dem, shift_contrib)
                            }
                            RayState::RecentlyConnected(ref s) => {
                                // Need to re-evaluate the incomming direction
                                // as the path are recently get connected
                                let shift_d_in_global = (s.its.p - main.its.p).normalize();
                                let shift_d_in_local = main.its.frame.to_local(shift_d_in_global);
                                if shift_d_in_local.z <= 0.0 || (!main_light_visible) {
                                    (0.0, Color::zero())
                                } else {
                                    assert!(!main.its.mesh.bsdf.is_smooth());

                                    // BSDF
                                    let shift_bsdf_pdf = f64::from(
                                        main.its
                                            .mesh
                                            .bsdf
                                            .pdf(
                                                &s.its.uv,
                                                &shift_d_in_local,
                                                &main_d_out_local,
                                                Domain::SolidAngle,
                                            )
                                            .value(),
                                    );
                                    let shift_bsdf_value = main.its.mesh.bsdf.eval(
                                        &s.its.uv,
                                        &shift_d_in_local,
                                        &main_d_out_local,
                                        Domain::SolidAngle,
                                    );
                                    // Compute and return
                                    let shift_weight_dem = (s.pdf / main.pdf).powi(MIS_POWER)
                                        * (main_light_pdf.powi(MIS_POWER)
                                            + shift_bsdf_pdf.powi(MIS_POWER));
                                    let shift_contrib =
                                        s.throughput * shift_bsdf_value * main_emitter_rad;
                                    (shift_weight_dem, shift_contrib)
                                }
                            }
                            RayState::NotConnected(ref s) => {
                                // Get intersection informations
                                let shift_hit_mesh = &s.its.mesh;
                                let intersectable_light = true;
                                let main_bsdf_rought = true;
                                let shift_bsdf_rought = !s.its.mesh.bsdf.is_smooth();

                                if !intersectable_light || (main_bsdf_rought && shift_bsdf_rought) {
                                    // Sample the light from the point
                                    let shift_light_record = emitters
                                        .sample_light(&s.its.p, r_sel_rand, r_rand, uv_rand);
                                    let shift_light_visible =
                                        accel.visible(&s.its.p, &shift_light_record.p);
                                    let shift_emitter_rad = if shift_light_visible {
                                        shift_light_record.weight
                                            * (shift_light_record.pdf.value()
                                                / main_light_record.pdf.value())
                                    } else {
                                        Color::zero()
                                    };
                                    let shift_d_out_local =
                                        s.its.frame.to_local(shift_light_record.d);
                                    // BSDF evaluation
                                    let shift_light_pdf = f64::from(shift_light_record.pdf.value());
                                    let shift_bsdf_value = shift_hit_mesh.bsdf.eval(
                                        &s.its.uv,
                                        &s.its.wi,
                                        &shift_d_out_local,
                                        Domain::SolidAngle, // Already check that we are on a non smooth surface
                                    );
                                    let shift_bsdf_pdf = if shift_light_visible {
                                        f64::from(
                                            shift_hit_mesh
                                                .bsdf
                                                .pdf(
                                                    &s.its.uv,
                                                    &s.its.wi,
                                                    &shift_d_out_local,
                                                    Domain::SolidAngle,
                                                )
                                                .value(),
                                        )
                                    } else {
                                        0.0
                                    };
                                    // Compute Jacobian: Here the ratio of geometry terms
                                    let jacobian = f64::from(
                                        (shift_light_record.n.dot(shift_light_record.d)
                                            * main_geom_dsquared)
                                            .abs()
                                            / (main_geom_cos_light
                                                * (s.its.p - shift_light_record.p).magnitude2())
                                            .abs(),
                                    );
                                    assert!(jacobian.is_finite());
                                    assert!(jacobian >= 0.0);
                                    // Bake the final results
                                    let shift_weight_dem = (jacobian * (s.pdf / main.pdf))
                                        .powi(MIS_POWER)
                                        * (shift_light_pdf.powi(MIS_POWER)
                                            + shift_bsdf_pdf.powi(MIS_POWER));
                                    let shift_contrib = (jacobian as f32)
                                        * s.throughput
                                        * shift_bsdf_value
                                        * shift_emitter_rad;
                                    (shift_weight_dem, shift_contrib)
                                } else {
                                    // In this case, we need to intersect directly the light source
                                    // however, the decision made inside GPT is to not handle this case
                                    // so we need to mark the shift failed
                                    (0.0, Color::zero())
                                }
                            }
                        };

                        if self.min_depth.map_or(true, |min| depth >= min) {
                            let weight =
                                (main_weight_num / (main_weight_dem + shift_weight_dem)) as f32;
                            assert!(weight.is_finite());
                            assert!(weight >= 0.0);
                            assert!(weight <= 1.0);
                            l_i.main += main_contrib * weight;
                            l_i.radiances[i] += shift_contrib * weight;
                            l_i.gradients[i] += (shift_contrib - main_contrib) * weight;
                        }
                    }
                }
            }

            /////////////////////////////////
            // BSDF sampling
            /////////////////////////////////
            // Compute an new direction (diffuse)
            let main_sampled_bsdf =
                match main
                    .its
                    .mesh
                    .bsdf
                    .sample(&main.its.uv, &main.its.wi, sampler.next2d())
                {
                    Some(x) => x,
                    None => return l_i,
                };

            // Generate the new ray and do the intersection
            let main_d_out_global = main.its.frame.to_world(main_sampled_bsdf.d);
            main.ray = Ray::new(main.its.p, main_d_out_global);
            let main_pred_its = main.its; // Need to save the previous hit
            main.its = match accel.trace(&main.ray) {
                Some(x) => x,
                None => return l_i,
            };
            let main_next_mesh = main.its.mesh;

            // Check that we have intersected a light or not
            let (main_light_pdf, main_emitter_rad) = {
                if main_next_mesh.is_light() && main.its.cos_theta() > 0.0 {
                    let light_pdf = f64::from(
                        emitters
                            .direct_pdf(main.its.mesh, &LightSamplingPDF::new(&main.ray, &main.its))
                            .value(),
                    );
                    (light_pdf, main_next_mesh.emission)
                } else {
                    (0.0, Color::zero())
                }
            };

            // Update the main path
            let main_pdf_pred = main.pdf;
            let main_bsdf_pdf = f64::from(main_sampled_bsdf.pdf.value());
            main.throughput *= &(main_sampled_bsdf.weight);
            main.pdf *= main_bsdf_pdf;
            // Check if we are in a correct state or not
            // Otherwise, kill the main process
            if main.pdf == 0.0 || main.throughput.is_zero() {
                return l_i;
            }

            let main_weight_num = main_bsdf_pdf.powi(MIS_POWER);
            let main_contrib = main.throughput * main_emitter_rad;

            offsets = offsets
                .into_iter()
                .enumerate()
                .map(|(i, offset)| {
                    struct ShiftResult<'a> {
                        pub weight_dem: f64,
                        pub contrib: Color,
                        pub state: RayState<'a>,
                        pub half_vector: bool,
                    };
                    impl<'a> Default for ShiftResult<'a> {
                        fn default() -> Self {
                            Self {
                                weight_dem: 0.0,
                                contrib: Color::zero(),
                                state: RayState::Dead,
                                half_vector: false,
                            }
                        }
                    };
                    let result: ShiftResult = match offset {
                        RayState::Dead => ShiftResult::default(),
                        RayState::Connected(mut s) => {
                            let shift_pdf_pred = s.pdf;
                            // Update the shifted path
                            s.throughput *= &(main_sampled_bsdf.weight);
                            s.pdf *= main_bsdf_pdf;
                            // Compute the return values
                            let shift_weight_dem = (shift_pdf_pred / main_pdf_pred).powi(MIS_POWER)
                                * (main_bsdf_pdf.powi(MIS_POWER) + main_light_pdf.powi(MIS_POWER));
                            let shift_contrib = s.throughput * main_emitter_rad;
                            ShiftResult {
                                weight_dem: shift_weight_dem,
                                contrib: shift_contrib,
                                state: RayState::Connected(s),
                                half_vector: false,
                            }
                        }
                        RayState::RecentlyConnected(mut s) => {
                            if main_pred_its.mesh.bsdf.is_smooth() {
                                ShiftResult::default()
                            } else {
                                let shift_d_in_global = (s.its.p - main.ray.o).normalize();
                                let shift_d_in_local =
                                    main_pred_its.frame.to_local(shift_d_in_global);
                                if shift_d_in_local.z <= 0.0 {
                                    // FIXME: Dead path as we do not deal with glass
                                    ShiftResult::default()
                                } else {
                                    // BSDF
                                    let shift_bsdf_pdf = f64::from(
                                        main_pred_its
                                            .mesh
                                            .bsdf
                                            .pdf(
                                                &main_pred_its.uv,
                                                &shift_d_in_local,
                                                &main_sampled_bsdf.d,
                                                Domain::SolidAngle,
                                            )
                                            .value(),
                                    );
                                    let shift_bsdf_value = main_pred_its.mesh.bsdf.eval(
                                        &main_pred_its.uv,
                                        &shift_d_in_local,
                                        &main_sampled_bsdf.d,
                                        Domain::SolidAngle,
                                    );
                                    // Update main path
                                    let shift_pdf_pred = s.pdf;
                                    s.throughput *= &(shift_bsdf_value / (main_bsdf_pdf as f32));
                                    s.pdf *= shift_bsdf_pdf;
                                    // Compute and return
                                    let shift_weight_dem = (shift_pdf_pred / main_pdf_pred)
                                        .powi(MIS_POWER)
                                        * (shift_bsdf_pdf.powi(MIS_POWER)
                                            + main_light_pdf.powi(MIS_POWER));
                                    let shift_contrib = s.throughput * main_emitter_rad;
                                    ShiftResult {
                                        weight_dem: shift_weight_dem,
                                        contrib: shift_contrib,
                                        state: RayState::Connected(s),
                                        half_vector: false,
                                    }
                                }
                            }
                        }
                        RayState::NotConnected(mut s) => {
                            let main_bsdf_rought = !main_pred_its.mesh.bsdf.is_smooth();
                            let main_next_bsdf_rought = !main_next_mesh.bsdf.is_smooth();
                            let shift_bsdf_rought = !s.its.mesh.bsdf.is_smooth();
                            if main_bsdf_rought && main_next_bsdf_rought && shift_bsdf_rought {
                                // In this case, we can do the reconnection
                                if !accel.visible(&s.its.p, &main.its.p) {
                                    ShiftResult::default()
                                } else {
                                    // Compute the ratio of geometry factors
                                    let shift_d_out_global = (main.its.p - s.its.p).normalize();
                                    let shift_d_out_local =
                                        s.its.frame.to_local(shift_d_out_global);
                                    let jacobian = f64::from(
                                        (main.its.n_g.dot(-shift_d_out_global)
                                            * main.its.dist.powi(2))
                                        .abs()
                                            / (main.its.n_g.dot(-main.ray.d)
                                                * (s.its.p - main.its.p).magnitude2())
                                            .abs(),
                                    );
                                    assert!(jacobian.is_finite());
                                    assert!(jacobian >= 0.0);
                                    // BSDF
                                    let shift_bsdf_value = s.its.mesh.bsdf.eval(
                                        &s.its.uv,
                                        &s.its.wi,
                                        &shift_d_out_local,
                                        Domain::SolidAngle, // Already checked that we are not on a smooth surface
                                    );
                                    let shift_bsdf_pdf = f64::from(
                                        s.its
                                            .mesh
                                            .bsdf
                                            .pdf(
                                                &s.its.uv,
                                                &s.its.wi,
                                                &shift_d_out_local,
                                                Domain::SolidAngle,
                                            )
                                            .value(),
                                    );
                                    // Update shift path
                                    let shift_pdf_pred = s.pdf;
                                    s.throughput *=
                                        &(shift_bsdf_value * (jacobian / main_bsdf_pdf) as f32);
                                    s.pdf *= shift_bsdf_pdf * jacobian;

                                    // Two case:
                                    // - the main are on a emitter, need to do MIS
                                    // - the main are not on a emitter, just do a reconnection
                                    let (shift_emitter_rad, shift_emitter_pdf) =
                                        if main_light_pdf == 0.0 {
                                            // The base path did not hit a light source
                                            // FIXME: Do not use the trick of 0 PDF
                                            (Color::zero(), 0.0)
                                        } else {
                                            let shift_emitter_pdf = emitters
                                                .direct_pdf(
                                                    main_next_mesh,
                                                    &LightSamplingPDF {
                                                        o: s.its.p,
                                                        p: main.its.p,
                                                        n: main.its.n_g,
                                                        dir: shift_d_out_global,
                                                    },
                                                )
                                                .value();
                                            // FIXME: We return without the cos as the light
                                            // FIXME: does not change, does it true for non uniform light?
                                            (main_emitter_rad, f64::from(shift_emitter_pdf))
                                        };

                                    // Return the shift path updated + MIS weights
                                    let shift_weight_dem = (shift_pdf_pred / main_pdf_pred)
                                        .powi(MIS_POWER)
                                        * (shift_bsdf_pdf.powi(MIS_POWER)
                                            + shift_emitter_pdf.powi(MIS_POWER));
                                    let shift_contrib = s.throughput * shift_emitter_rad;
                                    ShiftResult {
                                        weight_dem: shift_weight_dem,
                                        contrib: shift_contrib,
                                        state: RayState::RecentlyConnected(s),
                                        half_vector: false,
                                    }
                                }
                            } else {
                                // The two BDSF are not both discrete BDSF
                                // in this case, we need to kill the shift mapping
                                // operation
                                let shift_success = !main_bsdf_rought && !shift_bsdf_rought;

                                // In this case, we need to continue to shift the offset path
                                // we do that using half-vector copy
                                // note that if the light is intersected, we add its contribution
                                // NOTE: In this case, we have both of the path that are on a delta surface
                                let (mut success, _jacobian, wo) = {
                                    // Half vector copy
                                    let tan_space_main_wi = main_pred_its.wi;
                                    let tan_space_main_wo = main_sampled_bsdf.d;
                                    let tan_space_shift_wi = s.its.wi;
                                    let main_eta = 1.0;
                                    let shift_eta = 1.0;
                                    if tan_space_main_wi.z * tan_space_main_wo.z < 0.0 {
                                        // Reflection
                                        if main_eta == 1.0 || shift_eta == 1.0 {
                                            // This will be null interaction
                                            // need to handle it properly
                                            (false, 1.0, Vector3::new(0.0, 0.0, 0.0))
                                        } else {
                                            let tan_space_hv_main_unorm = if tan_space_main_wi.z
                                                < 0.0
                                            {
                                                -(tan_space_main_wi * main_eta + tan_space_main_wo)
                                            } else {
                                                -(tan_space_main_wi + main_eta * tan_space_main_wo)
                                            };
                                            let tan_space_hv_main =
                                                tan_space_hv_main_unorm.normalize();
                                            let tan_space_shift_wo = reflect_vector(
                                                tan_space_shift_wi,
                                                tan_space_hv_main,
                                            );
                                            if tan_space_shift_wo.x == 0.0
                                                && tan_space_shift_wo.y == 0.0
                                                && tan_space_shift_wo.z == 0.0
                                            {
                                                (false, 1.0, Vector3::new(0.0, 0.0, 0.0))
                                            } else {
                                                let tan_space_hv_shift_unorm =
                                                    if tan_space_shift_wi.z < 0.0 {
                                                        -(tan_space_shift_wi * shift_eta
                                                            + tan_space_shift_wo)
                                                    } else {
                                                        -(tan_space_shift_wi
                                                            + shift_eta * tan_space_shift_wo)
                                                    };
                                                let lenght_sqr = tan_space_hv_shift_unorm
                                                    .magnitude2()
                                                    / (tan_space_hv_main_unorm.magnitude2());
                                                let wo_dot_hv = tan_space_main_wo
                                                    .dot(tan_space_hv_main)
                                                    / tan_space_shift_wo.dot(tan_space_hv_main);
                                                (
                                                    true,
                                                    lenght_sqr * wo_dot_hv.abs(),
                                                    tan_space_shift_wo,
                                                )
                                            }
                                        }
                                    } else {
                                        // Reflexion
                                        let tan_space_hv_main =
                                            (tan_space_main_wo + tan_space_main_wi).normalize();
                                        let tan_space_shift_wo =
                                            reflect_vector(tan_space_shift_wi, tan_space_hv_main);
                                        let wo_dot_h = tan_space_shift_wo.dot(tan_space_hv_main)
                                            / tan_space_main_wo.dot(tan_space_hv_main);
                                        (true, wo_dot_h.abs(), tan_space_shift_wo)
                                    }
                                };
                                let jacobian = 1.0; // TODO: Always dirac
                                success &= shift_success; // TODO: Due to Rust lang return policy. Check how to exist to closure with return

                                if !success {
                                    let mut result = ShiftResult::default();
                                    result.half_vector = true;
                                    result
                                } else {
                                    // Pre-mult with the Jacobian
                                    s.throughput *= jacobian;
                                    s.pdf *= f64::from(jacobian);
                                    // Evaluate the new direction
                                    s.throughput *= &s.its.mesh.bsdf.eval(
                                        &s.its.uv,
                                        &s.its.wi,
                                        &wo,
                                        Domain::Discrete,
                                    );
                                    s.pdf *= f64::from(
                                        s.its
                                            .mesh
                                            .bsdf
                                            .pdf(&s.its.uv, &s.its.wi, &wo, Domain::Discrete)
                                            .value(),
                                    );
                                    // Shoot a ray to compute the next intersection
                                    let shift_d_out_global = s.its.frame.to_world(wo);
                                    s.ray = Ray::new(s.its.p, shift_d_out_global);
                                    let new_its = accel.trace(&s.ray);
                                    if let Some(new_its) = new_its {
                                        s.its = new_its;
                                        let shift_emitter_rad = if s.its.mesh.is_light() {
                                            s.its.mesh.emission
                                        } else {
                                            Color::zero()
                                        };
                                        ShiftResult {
                                            weight_dem: s.pdf,
                                            contrib: s.throughput * shift_emitter_rad,
                                            state: RayState::NotConnected(s),
                                            half_vector: true,
                                        }
                                    } else {
                                        let mut result = ShiftResult::default();
                                        result.half_vector = true;
                                        result
                                    }
                                }
                            }
                        }
                    };

                    // Due to the shift mapping design choice
                    // We do not use MIS when have hit the light
                    // when the shift path is not reconnected
                    let main_weight_dem = if result.half_vector {
                        main_bsdf_pdf.powi(MIS_POWER)
                    } else {
                        main_bsdf_pdf.powi(MIS_POWER) + main_light_pdf.powi(MIS_POWER)
                    };

                    // Update the contributions
                    if self.min_depth.map_or(true, |min| depth >= min) {
                        let weight =
                            (main_weight_num / (main_weight_dem + result.weight_dem)) as f32;
                        assert!(weight.is_finite());
                        assert!(weight >= 0.0);
                        assert!(weight <= 1.0);
                        l_i.main += main_contrib * weight;
                        l_i.radiances[i] += result.contrib * weight;
                        l_i.gradients[i] += (result.contrib - main_contrib) * weight;
                    }
                    // Return the new state
                    result.state
                })
                .collect::<Vec<RayState>>();

            // Russian roulette
            let rr_pdf = main.throughput.channel_max().min(0.95);
            if rr_pdf < sampler.next() {
                break;
            }
            main.throughput /= rr_pdf;
            offsets
                .iter_mut()
                .for_each(|o| o.apply_russian_roulette(rr_pdf));

            // Increase the depth of the current path
            depth += 1;
        }

        l_i
    }
}
