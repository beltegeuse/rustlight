use cgmath::*;
use integrators::gradient_domain::*;
use integrators::path::*;
use integrators::*;
use scene::*;
use structure::*;

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
            &mut RayState::Dead => {}
            &mut RayState::NotConnected(ref mut e)
            | &mut RayState::Connected(ref mut e)
            | &mut RayState::RecentlyConnected(ref mut e) => {
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

impl Integrator<ColorGradient> for IntegratorPath {
    fn compute<S: Sampler>(
        &self,
        (ix, iy): (u32, u32),
        scene: &Scene,
        sampler: &mut S,
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
                let main_light_pdf = main_light_record.pdf.value() as f64;
                let main_bsdf_value = main.its.mesh.bsdf.eval(
                    &main.its.uv,
                    &main.its.wi,
                    &main_d_out_local,
                ); // f(...) * cos(...)
                let main_bsdf_pdf = if main_light_visible {
                    main.its
                        .mesh
                        .bsdf
                        .pdf(&main.its.uv, &main.its.wi, &main_d_out_local)
                        .value() as f64
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
                            &RayState::Dead => {
                                (main_weight_num / (0.0001 + main_weight_dem), Color::zero())
                            }
                            &RayState::Connected(ref s) => {
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
                            &RayState::RecentlyConnected(ref s) => {
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
                                        main.its
                                            .mesh
                                            .bsdf
                                            .pdf(&s.its.uv, &shift_d_in_local, &main_d_out_local)
                                            .value() as f64;
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
                            &RayState::NotConnected(ref s) => {
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
                                let shift_light_pdf = shift_light_record.pdf.value() as f64;
                                let shift_bsdf_value = shift_hit_mesh.bsdf.eval(
                                    &s.its.uv,
                                    &s.its.wi,
                                    &shift_d_out_local,
                                );
                                let shift_bsdf_pdf = if shift_light_visible {
                                    shift_hit_mesh
                                        .bsdf
                                        .pdf(&s.its.uv, &s.its.wi, &shift_d_out_local)
                                        .value() as f64
                                } else {
                                    0.0
                                };
                                // Compute Jacobian: Here the ratio of geometry terms
                                let jacobian = ((shift_light_record.n.dot(shift_light_record.d)
                                    * main_geom_dsquared)
                                    .abs()
                                    / (main_geom_cos_light
                                        * (s.its.p - shift_light_record.p).magnitude2())
                                        .abs())
                                    as f64;
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
                            l_i.main += main_contrib; // * weight;
                                                      //l_i.radiances[i] += shift_contrib * weight;
                            l_i.gradients[i] += (shift_contrib - main_contrib) * weight;
                        }
                    }
                }
            }

            /////////////////////////////////
            // BSDF sampling
            /////////////////////////////////
            // Compute an new direction (diffuse)
            let main_sampled_bsdf = match main.its.mesh.bsdf.sample(
                &main.its.uv,
                &main.its.wi,
                sampler.next2d(),
            ) {
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
                    let light_pdf = scene
                        .direct_pdf(LightSamplingPDF::new(&main.ray, &main.its))
                        .value() as f64;
                    (light_pdf, main_next_mesh.emission.clone())
                } else {
                    (0.0, Color::zero())
                }
            };

            // Update the main path
            let main_pdf_pred = main.pdf;
            let main_bsdf_pdf = main_sampled_bsdf.pdf.value() as f64;
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
                                let shift_bsdf_pdf = main_pred_its
                                    .mesh
                                    .bsdf
                                    .pdf(&main_pred_its.uv, &shift_d_in_local, &main_sampled_bsdf.d)
                                    .value()
                                    as f64;
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
                                let jacobian = ((main.its.n_g.dot(-shift_d_out_global)
                                    * main.its.dist.powi(2))
                                    .abs()
                                    / (main.its.n_g.dot(-main.ray.d)
                                        * (s.its.p - main.its.p).magnitude2())
                                        .abs())
                                    as f64;
                                assert!(jacobian.is_finite());
                                assert!(jacobian >= 0.0);
                                // BSDF
                                let shift_bsdf_value = s.its.mesh.bsdf.eval(
                                    &s.its.uv,
                                    &s.its.wi,
                                    &shift_d_out_local,
                                );
                                let shift_bsdf_pdf =
                                    s.its
                                        .mesh
                                        .bsdf
                                        .pdf(&s.its.uv, &s.its.wi, &shift_d_out_local)
                                        .value() as f64;
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
                                    (main_emitter_rad.clone(), shift_emitter_pdf as f64)
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
                        l_i.main += main_contrib; // * weight;
                                                  //l_i.radiances[i] += shift_contrib * weight;
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
