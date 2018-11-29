use cgmath::*;
use integrators::*;
use structure::*;

pub struct IntegratorDirect {
    pub nb_bsdf_samples: u32,
    pub nb_light_samples: u32,
}

impl Integrator for IntegratorDirect {
    fn compute(&mut self, scene: &Scene) -> Bitmap {
        compute_mc(self, scene)
    }
}
impl IntegratorMC for IntegratorDirect {
    fn compute_pixel(&self, (ix, iy): (u32, u32), scene: &Scene, sampler: &mut Sampler) -> Color {
        let pix = Point2::new(ix as f32 + sampler.next(), iy as f32 + sampler.next());
        let ray = scene.camera.generate(pix);
        let mut l_i = Color::zero();

        // Do the intersection for the first path
        let its = match scene.trace(&ray) {
            Some(its) => its,
            None => return l_i,
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
            let light_record =
                scene.sample_light(&its.p, sampler.next(), sampler.next(), sampler.next2d());
            let light_pdf = match light_record.pdf {
                PDF::SolidAngle(v) => v,
                _ => panic!("Wrong light PDF"),
            };

            let d_out_local = its.frame.to_local(light_record.d);
            if light_record.is_valid()
                && scene.visible(&its.p, &light_record.p)
                && d_out_local.z > 0.0
            {
                // Compute the contribution of direct lighting
                // FIXME: A bit waste full, need to detect before sampling the light...
                if let PDF::SolidAngle(pdf_bsdf) = its.mesh.bsdf.pdf(&its.uv, &its.wi, &d_out_local, Domain::SolidAngle)
                {
                    // Compute MIS weights
                    let weight_light =
                        mis_weight(light_pdf * weight_nb_light, pdf_bsdf * weight_nb_bsdf);
                    l_i += &(weight_light
                        * its.mesh.bsdf.eval(&its.uv, &its.wi, &d_out_local, Domain::SolidAngle)
                        * weight_nb_light
                        * light_record.weight);
                }
            }
        }

        /////////////////////////////////
        // BSDF sampling
        /////////////////////////////////
        // Compute an new direction (diffuse)
        for _ in 0..self.nb_bsdf_samples {
            if let Some(sampled_bsdf) = its.mesh.bsdf.sample(&its.uv, &its.wi, sampler.next2d()) {
                // Generate the new ray and do the intersection
                let d_out_world = its.frame.to_world(sampled_bsdf.d);
                let ray = Ray::new(its.p, d_out_world);
                let next_its = match scene.trace(&ray) {
                    Some(x) => x,
                    None => continue,
                };

                // Check that we have intersected a light or not
                if next_its.mesh.is_light() && next_its.cos_theta() > 0.0 {
                    let weight_bsdf = match sampled_bsdf.pdf {
                        PDF::SolidAngle(bsdf_pdf) => {
                            let light_pdf = scene
                                .direct_pdf(&LightSamplingPDF::new(&ray, &next_its))
                                .value();
                            mis_weight(bsdf_pdf * weight_nb_bsdf, light_pdf * weight_nb_light)
                        }
                        PDF::Discrete(_v) => 1.0,
                        _ => {
                            warn!("Wrong PDF values retrieve on an intersected mesh");
                            continue;
                        }
                    };

                    l_i +=
                        weight_bsdf * sampled_bsdf.weight * next_its.mesh.emission * weight_nb_bsdf;
                }
            }
        }

        l_i
    }
}
