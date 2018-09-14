use cgmath::*;
use integrators::gradient::*;
use integrators::*;
use structure::*;

pub struct IntegratorGradientPath {
    pub max_depth: Option<u32>,
    pub min_depth: Option<u32>,
    pub recons: Box<PoissonReconstruction + Sync>,
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
            RayState::NotConnected(e) => if e.its.cos_theta() <= 0.0 {
                RayState::Dead
            } else {
                RayState::NotConnected(e)
            },
            RayState::RecentlyConnected(e) => if e.its.n_s.dot(e.ray.d) > 0.0 {
                RayState::Dead
            } else {
                RayState::RecentlyConnected(e)
            },
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

    pub fn new((x, y): (f32, f32), off: &Point2<i32>, scene: &'a Scene) -> RayState<'a> {
        let pix = Point2::new(x + off.x as f32, y + off.y as f32);
        if pix.x < 0.0
            || pix.x > (scene.camera.size().x as f32)
            || pix.y < 0.0
            || pix.y > (scene.camera.size().y as f32)
        {
            return RayState::Dead;
        }

        let ray = scene.camera.generate(pix);
        let its = match scene.trace(&ray) {
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
    fn reconstruct(&self) -> &Box<PoissonReconstruction + Sync> {
        &self.recons
    }

    fn compute_gradients(&mut self, scene: &Scene) -> Bitmap {
        // The buffers names are always:
        // ["very_direct", ("primal", "gradient_x", "gradient_y")+]
        let (nb_buffers, buffernames) =
            if let Some(number_buffers) = self.recons.need_variance_estimates() {
                let mut buffernames = Vec::new();
                buffernames.reserve((3 * number_buffers) + 1);
                buffernames.push(String::from("very_direct"));
                for i in 0..number_buffers {
                    buffernames.push(format!("primal_{}", i));
                    buffernames.push(format!("gradient_x_{}", i));
                    buffernames.push(format!("gradient_y_{}", i));
                }
                (number_buffers, buffernames)
            } else {
                (
                    1,
                    vec![
                        String::from("very_direct"),
                        String::from("primal"),
                        String::from("gradient_x"),
                        String::from("gradient_y"),
                    ],
                )
            };

        let id_very_direct = 0;
        let id_primal = 1;
        let id_gradient_x = 2;
        let id_gradient_y = 3;

        // Compare to path tracing, make the block a bit bigger
        // so that we can store all the path contribution
        struct BlockInfo {
            pub x_pos_off: u32,
            pub y_pos_off: u32,
            pub x_size_off: u32,
            pub y_size_off: u32,
        };
        let mut image_blocks = Vec::new();
        for ix in StepRangeInt::new(0, scene.camera.size().x as usize, 16) {
            for iy in StepRangeInt::new(0, scene.camera.size().y as usize, 16) {
                let pos_off = Point2 {
                    x: cmp::max(0, ix as i32 - 1) as u32,
                    y: cmp::max(0, iy as i32 - 1) as u32,
                };
                let desired_size = Vector2 {
                    x: 16 + if ix == 0 { 1 } else { 2 },
                    y: 16 + if iy == 0 { 1 } else { 2 },
                };
                let max_size = Vector2 {
                    x: (scene.camera.size().x - pos_off.x) as u32,
                    y: (scene.camera.size().y - pos_off.y) as u32,
                };
                let mut block = Bitmap::new(
                    pos_off,
                    Vector2 {
                        x: cmp::min(desired_size.x, max_size.x),
                        y: cmp::min(desired_size.y, max_size.y),
                    },
                    &buffernames,
                );
                let info = BlockInfo {
                    x_pos_off: if ix == 0 { 0 } else { 1 },
                    y_pos_off: if iy == 0 { 0 } else { 1 },
                    x_size_off: if desired_size.x <= max_size.x { 1 } else { 0 },
                    y_size_off: if desired_size.y <= max_size.y { 1 } else { 0 },
                };
                image_blocks.push((info, block));
            }
        }

        let progress_bar = Mutex::new(ProgressBar::new(image_blocks.len() as u64));
        let pool = generate_pool(scene);
        pool.install(|| {
            image_blocks.par_iter_mut().for_each(|(info, im_block)| {
                let mut sampler = independent::IndependentSampler::default();
                for ix in info.x_pos_off..im_block.size.x - info.x_size_off {
                    for iy in info.y_pos_off..im_block.size.y - info.y_size_off {
                        for n in 0..scene.nb_samples() {
                            let c = self.compute_pixel(
                                (ix + im_block.pos.x, iy + im_block.pos.y),
                                scene,
                                &mut sampler,
                            );
                            // Accumulate the values inside the buffer
                            let pos = Point2::new(ix, iy);
                            let offset_buffers = (n % nb_buffers) * 3; // 3 buffers are in multiple version
                            im_block.accumulate(
                                pos,
                                c.main,
                                &buffernames[id_primal + offset_buffers],
                            );
                            im_block.accumulate(
                                pos,
                                c.very_direct,
                                &buffernames[id_very_direct].to_owned(),
                            );
                            for i in 0..4 {
                                // primal reuse
                                let off = GRADIENT_ORDER[i];
                                let pos_off = Point2::new(ix as i32 + off.x, iy as i32 + off.y);
                                im_block.accumulate_safe(
                                    pos_off,
                                    c.radiances[i],
                                    &buffernames[id_primal + offset_buffers],
                                );
                                // gradient
                                match GRADIENT_DIRECTION[i] {
                                    GradientDirection::X(v) => match v {
                                        1 => im_block.accumulate(
                                            pos,
                                            c.gradients[i],
                                            &buffernames[id_gradient_x + offset_buffers],
                                        ),
                                        -1 => im_block.accumulate_safe(
                                            pos_off,
                                            c.gradients[i] * -1.0,
                                            &buffernames[id_gradient_x + offset_buffers],
                                        ),
                                        _ => panic!("wrong displacement X"), // FIXME: Fix the enum
                                    },
                                    GradientDirection::Y(v) => match v {
                                        1 => im_block.accumulate(
                                            pos,
                                            c.gradients[i],
                                            &buffernames[id_gradient_y + offset_buffers],
                                        ),
                                        -1 => im_block.accumulate_safe(
                                            pos_off,
                                            c.gradients[i] * -1.0,
                                            &buffernames[id_gradient_y + offset_buffers],
                                        ),
                                        _ => panic!("wrong displacement Y"),
                                    },
                                }
                            }
                        }
                    }
                }
                im_block.scale(1.0 / (scene.nb_samples() as f32));
                // Renormalize correctly the buffer informations
                for i in 0..nb_buffers {
                    let offset_buffers = i * 3; // 3 buffer that have multiple entries
                                                // 4 strategies as reuse primal
                    im_block.scale_buffer(
                        0.25 * nb_buffers as f32,
                        &buffernames[id_primal + offset_buffers],
                    );
                    im_block.scale_buffer(
                        nb_buffers as f32,
                        &buffernames[id_gradient_x + offset_buffers],
                    );
                    im_block.scale_buffer(
                        nb_buffers as f32,
                        &buffernames[id_gradient_y + offset_buffers],
                    );
                }

                {
                    progress_bar.lock().unwrap().inc();
                }
            });
        });

        // Fill the image & do the reconstruct
        let mut image = Bitmap::new(Point2::new(0, 0), *scene.camera.size(), &buffernames);
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
        scene: &Scene,
        sampler: &mut Sampler,
    ) -> ColorGradient {
        let mut l_i = ColorGradient::default();
        let pix = (ix as f32 + sampler.next(), iy as f32 + sampler.next());
        let mut main = match RayState::new(pix, &Point2::new(0, 0), scene) {
            RayState::NotConnected(x) => x,
            _ => return l_i,
        };
        let mut offsets: Vec<RayState> = {
            GRADIENT_ORDER
                .iter()
                .map(|e| RayState::new(pix, e, &scene))
                .collect()
        };

        const MIS_POWER: i32 = 2;

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
            {
                let (r_sel_rand, r_rand, uv_rand) =
                    (sampler.next(), sampler.next(), sampler.next2d());
                let main_light_record =
                    scene.sample_light(&main.its.p, r_sel_rand, r_rand, uv_rand);
                let main_light_visible = scene.visible(&main.its.p, &main_light_record.p);
                let main_emitter_rad = if main_light_visible {
                    main_light_record.weight
                } else {
                    Color::zero()
                };
                let main_d_out_local = main.its.frame.to_local(main_light_record.d);
                // Evaluate BSDF values and light values
                let main_light_pdf = f64::from(main_light_record.pdf.value());
                let main_bsdf_value =
                    main.its
                        .mesh
                        .bsdf
                        .eval(&main.its.uv, &main.its.wi, &main_d_out_local); // f(...) * cos(...)
                let main_bsdf_pdf = if main_light_visible {
                    f64::from(main.its
                        .mesh
                        .bsdf
                        .pdf(&main.its.uv, &main.its.wi, &main_d_out_local)
                        .value())
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
                    // FIXME: Double check this condition. Normally, it will be fine
                    // FIXME: But pdf = 0 for the main path does not necessary imply
                    // FIXME: 0 probability for the shift path, no?
                    for (i, offset) in offsets.iter().enumerate() {
                        let (shift_weight_dem, shift_contrib) = match offset {
                            RayState::Dead => {
                                (main_weight_num / (0.0001 + main_weight_dem), Color::zero())
                            }
                            RayState::Connected(ref s) => {
                                // FIXME: See if we can simplify the structure, as we need to know:
                                //  - throughput
                                //  - pdf
                                // only
                                let shift_weight_dem = (s.pdf / main.pdf).powi(MIS_POWER)
                                    * (main_light_pdf.powi(MIS_POWER)
                                        + main_bsdf_pdf.powi(MIS_POWER));
                                let shift_contrib =
                                    s.throughput * main_bsdf_value * main_emitter_rad;
                                (shift_weight_dem, shift_contrib)
                            }
                            RayState::RecentlyConnected(ref s) => {
                                // Need to re-evaluate the BSDF as the incomming direction is different
                                // FIXME: We only need to know:
                                //  - throughput
                                //  - pdf
                                //  - incomming direction (in world space)
                                let shift_d_in_global = (s.its.p - main.its.p).normalize();
                                let shift_d_in_local = main.its.frame.to_local(shift_d_in_global);
                                if shift_d_in_local.z <= 0.0 || (!main_light_visible) {
                                    (0.0, Color::zero())
                                } else {
                                    // BSDF
                                    let shift_bsdf_pdf =
                                        f64::from(main.its
                                            .mesh
                                            .bsdf
                                            .pdf(&s.its.uv, &shift_d_in_local, &main_d_out_local)
                                            .value());
                                    let shift_bsdf_value = main.its.mesh.bsdf.eval(
                                        &s.its.uv,
                                        &shift_d_in_local,
                                        &main_d_out_local,
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
                                // FIXME: We need to check the light source type in order to continue or not
                                // FIXME: the ray tracing...

                                // Sample the light from the point
                                let shift_light_record =
                                    scene.sample_light(&s.its.p, r_sel_rand, r_rand, uv_rand);
                                let shift_light_visible =
                                    scene.visible(&s.its.p, &shift_light_record.p);
                                let shift_emitter_rad = if shift_light_visible {
                                    shift_light_record.weight
                                        * (shift_light_record.pdf.value()
                                            / main_light_record.pdf.value())
                                } else {
                                    Color::zero()
                                };
                                let shift_d_out_local = s.its.frame.to_local(shift_light_record.d);
                                // BSDF
                                let shift_light_pdf = f64::from(shift_light_record.pdf.value());
                                let shift_bsdf_value = shift_hit_mesh.bsdf.eval(
                                    &s.its.uv,
                                    &s.its.wi,
                                    &shift_d_out_local,
                                );
                                let shift_bsdf_pdf = if shift_light_visible {
                                    f64::from(shift_hit_mesh
                                        .bsdf
                                        .pdf(&s.its.uv, &s.its.wi, &shift_d_out_local)
                                        .value())
                                } else {
                                    0.0
                                };
                                // Compute Jacobian: Here the ratio of geometry terms
                                let jacobian = f64::from((shift_light_record.n.dot(shift_light_record.d)
                                    * main_geom_dsquared)
                                    .abs()
                                    / (main_geom_cos_light
                                        * (s.its.p - shift_light_record.p).magnitude2()).abs());
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
            main.its = match scene.trace(&main.ray) {
                Some(x) => x,
                None => return l_i,
            };
            let main_next_mesh = main.its.mesh;

            // Check that we have intersected a light or not
            let (main_light_pdf, main_emitter_rad) = {
                if main_next_mesh.is_light() && main.its.cos_theta() > 0.0 {
                    let light_pdf = f64::from(scene
                        .direct_pdf(LightSamplingPDF::new(&main.ray, &main.its))
                        .value());
                    (light_pdf, main_next_mesh.emission.clone())
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
            let main_weight_dem = main_bsdf_pdf.powi(MIS_POWER) + main_light_pdf.powi(MIS_POWER);
            let main_contrib = main.throughput * main_emitter_rad;

            offsets = offsets
                .into_iter()
                .enumerate()
                .map(|(i, offset)| {
                    let (shift_weight_dem, shift_contrib, new_state) = match offset {
                        RayState::Dead => (0.0, Color::zero(), RayState::Dead),
                        RayState::Connected(mut s) => {
                            let shift_pdf_pred = s.pdf;
                            // Update the shifted path
                            s.throughput *= &(main_sampled_bsdf.weight);
                            s.pdf *= main_bsdf_pdf;
                            // Compute the return values
                            let shift_weight_dem = (shift_pdf_pred / main_pdf_pred).powi(MIS_POWER)
                                * (main_bsdf_pdf.powi(MIS_POWER) + main_light_pdf.powi(MIS_POWER));
                            let shift_contrib = s.throughput * main_emitter_rad;
                            (shift_weight_dem, shift_contrib, RayState::Connected(s))
                        }
                        RayState::RecentlyConnected(mut s) => {
                            let shift_d_in_global = (s.its.p - main.ray.o).normalize();
                            let shift_d_in_local = main_pred_its.frame.to_local(shift_d_in_global);
                            if shift_d_in_local.z <= 0.0 {
                                // FIXME: Dead path as we do not deal with glass
                                (0.0, Color::zero(), RayState::Dead)
                            } else {
                                // BSDF
                                let shift_bsdf_pdf = f64::from(main_pred_its
                                    .mesh
                                    .bsdf
                                    .pdf(&main_pred_its.uv, &shift_d_in_local, &main_sampled_bsdf.d)
                                    .value());
                                let shift_bsdf_value = main_pred_its.mesh.bsdf.eval(
                                    &main_pred_its.uv,
                                    &shift_d_in_local,
                                    &main_sampled_bsdf.d,
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
                                (shift_weight_dem, shift_contrib, RayState::Connected(s))
                            }
                        }
                        RayState::NotConnected(mut s) => {
                            // FIXME: Always do a reconnection here
                            // FIXME: Implement half-vector copy
                            if !scene.visible(&s.its.p, &main.its.p) {
                                (0.0, Color::zero(), RayState::Dead) // FIXME: Found a way to do it in an elegant way
                            } else {
                                // Compute the ratio of geometry factors
                                let shift_d_out_global = (main.its.p - s.its.p).normalize();
                                let shift_d_out_local = s.its.frame.to_local(shift_d_out_global);
                                let jacobian = f64::from((main.its.n_g.dot(-shift_d_out_global)
                                    * main.its.dist.powi(2))
                                    .abs()
                                    / (main.its.n_g.dot(-main.ray.d)
                                        * (s.its.p - main.its.p).magnitude2())
                                        .abs());
                                assert!(jacobian.is_finite());
                                assert!(jacobian >= 0.0);
                                // BSDF
                                let shift_bsdf_value = s.its.mesh.bsdf.eval(
                                    &s.its.uv,
                                    &s.its.wi,
                                    &shift_d_out_local,
                                );
                                let shift_bsdf_pdf =
                                    f64::from(s.its
                                        .mesh
                                        .bsdf
                                        .pdf(&s.its.uv, &s.its.wi, &shift_d_out_local)
                                        .value());
                                // Update shift path
                                let shift_pdf_pred = s.pdf;
                                s.throughput *=
                                    &(shift_bsdf_value * (jacobian / main_bsdf_pdf) as f32);
                                s.pdf *= shift_bsdf_pdf * jacobian;

                                // Two case:
                                // - the main are on a emitter, need to do MIS
                                // - the main are not on a emitter, just do a reconnection
                                let (shift_emitter_rad, shift_emitter_pdf) = if main_light_pdf
                                    == 0.0
                                {
                                    // The base path did not hit a light source
                                    // FIXME: Do not use the trick of 0 PDF
                                    (Color::zero(), 0.0)
                                } else {
                                    let shift_emitter_pdf = scene
                                        .direct_pdf(LightSamplingPDF {
                                            mesh: main_next_mesh,
                                            o: s.its.p,
                                            p: main.its.p,
                                            n: main.its.n_g,
                                            dir: shift_d_out_global,
                                        })
                                        .value();
                                    // FIXME: We return without the cos as the light
                                    // FIXME: does not change, does it true for non uniform light?
                                    (main_emitter_rad.clone(), f64::from(shift_emitter_pdf))
                                };

                                // Return the shift path updated + MIS weights
                                let shift_weight_dem = (shift_pdf_pred / main_pdf_pred)
                                    .powi(MIS_POWER)
                                    * (shift_bsdf_pdf.powi(MIS_POWER)
                                        + shift_emitter_pdf.powi(MIS_POWER));
                                let shift_contrib = s.throughput * shift_emitter_rad;
                                (
                                    shift_weight_dem,
                                    shift_contrib,
                                    RayState::RecentlyConnected(s),
                                )
                            }
                        }
                    };
                    // Update the contributions
                    if self.min_depth.map_or(true, |min| depth >= min) {
                        //let weight = if shift_weight_dem == 0.0 { 1.0 } else { 0.5 };
                        let weight =
                            (main_weight_num / (main_weight_dem + shift_weight_dem)) as f32;
                        assert!(weight.is_finite());
                        assert!(weight >= 0.0);
                        assert!(weight <= 1.0);
                        l_i.main += main_contrib * weight;
                        l_i.radiances[i] += shift_contrib * weight;
                        l_i.gradients[i] += (shift_contrib - main_contrib) * weight;
                    }
                    // Return the new state
                    new_state
                })
                .collect();

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
