use crate::accel::*;
use crate::integrators::explicit::plane_single::*;
use crate::integrators::*;
use crate::math::*;
use crate::volume::*;
use cgmath::{InnerSpace, Point2};

pub struct IntegratorSinglePlaneUncorrelated {
    pub nb_primitive: usize,
    pub strategy: SinglePlaneStrategy,
}

impl Integrator for IntegratorSinglePlaneUncorrelated {
    fn compute(
        &mut self,
        sampler: &mut dyn Sampler,
        accel: &dyn Acceleration,
        scene: &Scene,
    ) -> BufferCollection {
        if scene.volume.is_none() {
            panic!("Volume integrator need a volume (add -m )");
        }
        // Extract the light source
        let emitters = scene
            .meshes
            .iter()
            .filter(|m| m.is_light())
            .collect::<Vec<_>>();

        let rect_lights = {
            emitters
                .iter()
                .map(|emitter| {
                    info!("Emitter vertices: {:?}", emitter.vertices);
                    info!("Emitter indices: {:?}", emitter.indices);
                    if emitter.vertices.len() != 3 && emitter.indices.len() != 2 {
                        panic!("Only support rectangular emitters");
                    }
                    RectangularLightSource::from_shape(emitter)
                })
                .collect::<Vec<_>>()
        };

        let generate_plane = |t: PlaneType,
                              light: &[RectangularLightSource],
                              id_emitter: usize,
                              sampler: &mut dyn Sampler,
                              m: &HomogenousVolume|
         -> SinglePhotonPlane {
            let d = {
                let mut d_out = cosine_sample_hemisphere(sampler.next2d());
                while d_out.z == 0.0 {
                    // Start to generate a new plane again
                    d_out = cosine_sample_hemisphere(sampler.next2d());
                }
                let frame = Frame::new(light[id_emitter].n);
                frame.to_world(d_out)
            };

            // FIXME: Faking position as it is not important for sampling the transmittance
            let ray_med = Ray::new(light[id_emitter].o, d);
            // TODO: Check if it is the code
            // Need to check the intersection distance iff need to failed ...
            // ray_med.tfar = intersection_distance;
            let mrec = m.sample(&ray_med, sampler.next());

            // Sample planes
            let sample = sampler.next2d();
            SinglePhotonPlane::new(
                t,
                &light[id_emitter],
                d,
                sample,
                sampler.next(),
                mrec.continued_t,
                id_emitter,
                m.sigma_s,
            )
        };

        // Generate the image block to get VPL efficiently
        let buffernames = vec![String::from("primal")];
        let mut image_blocks = generate_img_blocks(scene, sampler, &buffernames);
        let m = scene.volume.as_ref().unwrap();

        // Gathering all planes
        info!("Gathering Single planes...");
        let progress_bar = Mutex::new(ProgressBar::new(image_blocks.len() as u64));
        let pool = generate_pool(scene);
        let phase_function = PhaseFunction::Isotropic();
        pool.install(|| {
            image_blocks
                .par_iter_mut()
                .for_each(|(im_block, sampler_ray)| {
                    for ix in 0..im_block.size.x {
                        for iy in 0..im_block.size.y {
                            for _ in 0..scene.nb_samples {
                                let (ix_c, iy_c) = (ix + im_block.pos.x, iy + im_block.pos.y);
                                let pix = Point2::new(
                                    ix_c as f32 + sampler_ray.next(),
                                    iy_c as f32 + sampler_ray.next(),
                                );
                                let mut ray = scene.camera.generate(pix);

                                // Get the max distance
                                let max_dist = match accel.trace(&ray) {
                                    Some(x) => x.dist,
                                    None => std::f32::MAX,
                                };
                                ray.tfar = max_dist;

                                // Now gather all planes
                                let mut c = Color::value(0.0);
                                for _id_plane in 0..self.nb_primitive {
                                    let id_emitter =
                                        (sampler_ray.next() * rect_lights.len() as f32) as usize;
                                    let plane = match self.strategy {
                                        SinglePlaneStrategy::UT => generate_plane(
                                            PlaneType::UT,
                                            &rect_lights,
                                            id_emitter,
                                            sampler_ray.as_mut(),
                                            m,
                                        ),
                                        SinglePlaneStrategy::VT => generate_plane(
                                            PlaneType::VT,
                                            &rect_lights,
                                            id_emitter,
                                            sampler_ray.as_mut(),
                                            m,
                                        ),
                                        SinglePlaneStrategy::UV => generate_plane(
                                            PlaneType::UV,
                                            &rect_lights,
                                            id_emitter,
                                            sampler_ray.as_mut(),
                                            m,
                                        ),
                                        SinglePlaneStrategy::DiscreteMIS
                                        | SinglePlaneStrategy::Average => {
                                            // Generate 3 planes
                                            let plane_type = sampler_ray.next();
                                            if plane_type < (1.0 / 3.0) {
                                                generate_plane(
                                                    PlaneType::UV,
                                                    &rect_lights,
                                                    id_emitter,
                                                    sampler_ray.as_mut(),
                                                    m,
                                                )
                                            } else if plane_type < (2.0 / 3.0) {
                                                generate_plane(
                                                    PlaneType::VT,
                                                    &rect_lights,
                                                    id_emitter,
                                                    sampler_ray.as_mut(),
                                                    m,
                                                )
                                            } else {
                                                generate_plane(
                                                    PlaneType::UT,
                                                    &rect_lights,
                                                    id_emitter,
                                                    sampler_ray.as_mut(),
                                                    m,
                                                )
                                            }
                                        }
                                        SinglePlaneStrategy::UAlpha
                                        | SinglePlaneStrategy::ContinousMIS => generate_plane(
                                            PlaneType::UAlphaT,
                                            &rect_lights,
                                            id_emitter,
                                            sampler_ray.as_mut(),
                                            m,
                                        ),
                                    };

                                    let plane_its = plane.intersection(&ray);
                                    if plane_its.is_none() {
                                        continue;
                                    }
                                    let plane_its = plane_its.unwrap();

                                    let p_hit = ray.o + ray.d * plane_its.t_cam;
                                    let p_light =
                                        plane.light_position(&rect_lights[id_emitter], &plane_its);
                                    if accel.visible(&p_hit, &p_light) {
                                        let transmittance = {
                                            let mut ray_tr = Ray::new(ray.o, ray.d);
                                            ray_tr.tfar = plane_its.t_cam;
                                            m.transmittance(ray_tr)
                                        };
                                        let rho = phase_function
                                            .eval(&(-ray.d), &(p_light - p_hit).normalize());
                                        let w = match self.strategy {
                                            SinglePlaneStrategy::UT
                                            | SinglePlaneStrategy::UV
                                            | SinglePlaneStrategy::VT
                                            | SinglePlaneStrategy::UAlpha
                                            | SinglePlaneStrategy::ContinousMIS => 1.0,
                                            SinglePlaneStrategy::Average => 1.0 / 3.0,
                                            SinglePlaneStrategy::DiscreteMIS => {
                                                // Need to compute all possible shapes
                                                let d = p_hit - p_light;
                                                // TODO: Not used
                                                let t_sampled = d.magnitude();
                                                let d = d / t_sampled;
                                                let planes = [
                                                    SinglePhotonPlane::new(
                                                        PlaneType::UV,
                                                        &rect_lights[id_emitter],
                                                        d,
                                                        plane.sample,
                                                        0.0,
                                                        t_sampled,
                                                        id_emitter,
                                                        m.sigma_s,
                                                    ),
                                                    SinglePhotonPlane::new(
                                                        PlaneType::UT,
                                                        &rect_lights[id_emitter],
                                                        d,
                                                        plane.sample,
                                                        0.0,
                                                        t_sampled,
                                                        id_emitter,
                                                        m.sigma_s,
                                                    ),
                                                    SinglePhotonPlane::new(
                                                        PlaneType::VT,
                                                        &rect_lights[id_emitter],
                                                        d,
                                                        plane.sample,
                                                        0.0,
                                                        t_sampled,
                                                        id_emitter,
                                                        m.sigma_s,
                                                    ),
                                                ];
                                                // FIXME: Normally this code is unecessary
                                                // 	As we can reuse the plane retrived.
                                                // 	However, it seems to have a miss match between photon planes
                                                //	contribution calculation.
                                                let debug_id = match plane.plane_type {
                                                    PlaneType::UV => 0,
                                                    PlaneType::UT => 1,
                                                    PlaneType::VT => 2,
                                                    _ => unimplemented!(),
                                                };

                                                planes[debug_id].contrib(&ray.d).avg().powi(-1)
                                                    / planes
                                                        .iter()
                                                        .map(|p| p.contrib(&ray.d).avg().powi(-1))
                                                        .sum::<f32>()
                                            }
                                        };
                                        //assert!(w > 0.0 && w <= 1.0);

                                        // UV planes are not importance sampled (position/direction)
                                        // For other primitive, there such importance sampled approach.
                                        let flux = match self.strategy {
                                            SinglePlaneStrategy::ContinousMIS => {
                                                // Here we use their integration from
                                                // Normally, all the jacobian simplifies
                                                // So it is why we need to have a special estimator
                                                let w_cmis = 1.0
                                                    / ((2.0 / std::f32::consts::PI)
                                                        * (rect_lights[id_emitter]
                                                            .u
                                                            .cross(plane.d1)
                                                            .dot(ray.d)
                                                            .powi(2)
                                                            + rect_lights[id_emitter]
                                                                .v
                                                                .cross(plane.d1)
                                                                .dot(ray.d)
                                                                .powi(2))
                                                        .sqrt());
                                                w_cmis * plane.weight
                                            }
                                            _ => plane.contrib(&ray.d),
                                        };
                                        c += w
                                            * rho
                                            * transmittance
                                            * m.sigma_s
                                            * flux
                                            * (rect_lights.len() as f32)
                                            * (1.0 / self.nb_primitive as f32);
                                    }
                                }

                                im_block.accumulate(
                                    Point2 { x: ix, y: iy },
                                    c,
                                    &"primal".to_owned(),
                                );
                            }
                        }
                    } // Image block
                    im_block.scale(1.0 / (scene.nb_samples as f32));
                    {
                        progress_bar.lock().unwrap().inc();
                    }
                });
        });

        let mut image =
            BufferCollection::new(Point2::new(0, 0), *scene.camera.size(), &buffernames);
        for (im_block, _) in &image_blocks {
            image.accumulate_bitmap(im_block);
        }
        image
    }
}
