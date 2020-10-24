use crate::emitter::*;
use crate::integrators::*;
use cgmath::InnerSpace;

pub struct IntegratorDirect {
    pub nb_bsdf_samples: u32,
    pub nb_light_samples: u32,
}

impl Integrator for IntegratorDirect {
    fn compute(
        &mut self,
        sampler: &mut dyn Sampler,
        accel: &dyn Acceleration,
        scene: &Scene,
    ) -> BufferCollection {
        compute_mc(self, sampler, accel, scene)
    }
}
impl IntegratorMC for IntegratorDirect {
    fn compute_pixel(
        &self,
        (ix, iy): (u32, u32),
        accel: &dyn Acceleration,
        scene: &Scene,
        sampler: &mut dyn Sampler,
    ) -> Color {
        let pix = Point2::new(ix as f32 + sampler.next(), iy as f32 + sampler.next());
        let ray = scene.camera.generate(pix);
        let mut l_i = Color::zero();

        // Do the intersection for the first path
        let its = match accel.trace(&ray) {
            Some(its) => its,
            None => return scene.enviroment_luminance(ray.d),
        };

        // FIXME: Will not work with glass
        // Check if we go the right orientation
        if its.cos_theta() <= 0.0 {
            return l_i;
        }

        // Add the emission for the light intersection
        l_i += &its.mesh.emission;

        // Precompute for mis weights
        let weight_nb_bsdf = if self.nb_bsdf_samples == 0 {
            0.0
        } else {
            1.0 / (self.nb_bsdf_samples as f32)
        };
        let weight_nb_light = if self.nb_light_samples == 0 {
            0.0
        } else {
            1.0 / (self.nb_light_samples as f32)
        };

        /////////////////////////////////
        // Light sampling
        /////////////////////////////////
        // Explict connect to the light source
        for _ in 0..self.nb_light_samples {
            let light_record = scene.emitters().sample_light(
                &its.p,
                sampler.next(),
                sampler.next(),
                sampler.next2d(),
            );

            // Compute the contribution of direct lighting
            let d_out_local = its.frame.to_local(light_record.d);
            if light_record.is_valid()
                && accel.visible(&its.p, &light_record.p)
                && !its.mesh.bsdf.bsdf_type().is_smooth()
            {
                let weight_light = {
                    let pdf_bsdf = its.mesh.bsdf.pdf(
                        &its.uv,
                        &its.wi,
                        &d_out_local,
                        Domain::SolidAngle,
                        Transport::Importance,
                    );
                    match (light_record.pdf, pdf_bsdf) {
                        (PDF::SolidAngle(pdf_light), PDF::SolidAngle(pdf_bsdf)) => {
                            mis_weight(pdf_light * weight_nb_light, pdf_bsdf * weight_nb_bsdf)
                        }
                        (PDF::Discrete(_), PDF::Discrete(_)) => {
                            panic!("Impossible that both are discrete")
                        }
                        (PDF::Discrete(_), _) => 1.0, // The light is discrete, MIS does not apply
                        _ => todo!(),
                    }
                };

                l_i += &(weight_light
                    * its.mesh.bsdf.eval(
                        &its.uv,
                        &its.wi,
                        &d_out_local,
                        Domain::SolidAngle,
                        Transport::Importance,
                    )
                    * weight_nb_light
                    * light_record.weight);
            }
        }

        /////////////////////////////////
        // BSDF sampling
        /////////////////////////////////
        // Compute an new direction (diffuse)
        for _ in 0..self.nb_bsdf_samples {
            if let Some(sampled_bsdf) =
                its.mesh
                    .bsdf
                    .sample(&its.uv, &its.wi, sampler.next2d(), Transport::Importance)
            {
                // Generate the new ray and do the intersection
                let d_out_world = its.frame.to_world(sampled_bsdf.d);
                let ray = Ray::new(its.p, d_out_world);
                match accel.trace(&ray) {
                    Some(next_its) => {
                        // Check that we have intersected a light or not
                        if next_its.mesh.is_light() && next_its.cos_theta() > 0.0 {
                            let weight_bsdf = match sampled_bsdf.pdf {
                                PDF::SolidAngle(bsdf_pdf) => {
                                    let light_pdf = scene
                                        .emitters()
                                        .direct_pdf(
                                            next_its.mesh,
                                            &LightSamplingPDF::new(&ray, &next_its),
                                        )
                                        .value();
                                    mis_weight(
                                        bsdf_pdf * weight_nb_bsdf,
                                        light_pdf * weight_nb_light,
                                    )
                                }
                                PDF::Discrete(_v) => 1.0,
                                _ => {
                                    warn!("Wrong PDF values retrieve on an intersected mesh");
                                    continue;
                                }
                            };

                            l_i += weight_bsdf
                                * sampled_bsdf.weight
                                * next_its.mesh.emission
                                * weight_nb_bsdf;
                        }
                    }
                    None => {
                        if scene.emitter_environment.is_some() {
                            let envmap = scene.emitter_environment.as_ref().unwrap();
                            // We have to compute MIS
                            let weight_bsdf = match sampled_bsdf.pdf {
                                PDF::SolidAngle(bsdf_pdf) => {
                                    let bsphere = envmap.bsphere.as_ref().unwrap();
                                    let t = bsphere.intersect(&ray);
                                    let t = t.unwrap();

                                    let p = ray.o + ray.d * t;
                                    let n = (bsphere.center - p).normalize();
                                    if let PDF::SolidAngle(light_pdf) = scene.emitters().direct_pdf(
                                        envmap.as_ref(),
                                        &LightSamplingPDF {
                                            o: ray.o,
                                            p,
                                            n,
                                            dir: ray.d,
                                        },
                                    ) {
                                        mis_weight(
                                            bsdf_pdf * weight_nb_bsdf,
                                            light_pdf * weight_nb_light,
                                        )
                                    } else {
                                        1.0
                                    }
                                }
                                PDF::Discrete(_v) => 1.0,
                                _ => {
                                    warn!("Wrong PDF values retrieve on an intersected mesh");
                                    continue;
                                }
                            };

                            l_i += weight_bsdf
                                * sampled_bsdf.weight
                                * scene.enviroment_luminance(ray.d)
                                * weight_nb_bsdf;
                        }
                    }
                };
            }
        }

        l_i
    }
}
