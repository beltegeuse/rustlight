use cgmath::*;
use structure::*;
use scene::*;
use integrators::*;

pub struct IntegratorPath {
    pub max_depth: Option<u32>,
    pub min_depth: Option<u32>,
}

impl Integrator<Color> for IntegratorPath {
    fn compute<S: Sampler>(&self, (ix, iy): (u32, u32), scene: &Scene, sampler: &mut S) -> Color {
        // Generate the first ray
        let pix = Point2::new(ix as f32 + sampler.next(), iy as f32 + sampler.next());
        let mut ray = scene.camera.generate(pix);
        let mut l_i = Color::zero();
        let mut throughput = Color::one();

        // Check if we have a intersection with the primary ray
        let mut its = match scene.trace(&ray) {
            Some(x) => x,
            None => return l_i,
        };

        let mut depth: u32 = 1;
        while self.max_depth.map_or(true, |max| depth < max) {
            // Check if we go the right orientation
            if its.cos_theta() <= 0.0 {
                return l_i;
            }

            // Add the emission for the light intersection
            if self.min_depth.map_or(true, |min| depth >= min) && depth == 1 {
                l_i += &(throughput * its.mesh.emission);
            }

            /////////////////////////////////
            // Light sampling
            /////////////////////////////////
            // Explict connect to the light source
            let light_record =
                scene.sample_light(&its.p, sampler.next(), sampler.next(), sampler.next2d());
            let light_pdf = match light_record.pdf {
                PDF::SolidAngle(v) => v,
                _ => panic!("Unsupported light, abord"),
            };

            let d_out_local = its.frame.to_local(light_record.d);
            if light_record.is_valid() && scene.visible(&its.p, &light_record.p)
                && d_out_local.z > 0.0
                {
                    // Compute the contribution of direct lighting
                    // FIXME: A bit waste full, need to detect before sampling the light...
                    if let PDF::SolidAngle(pdf_bsdf) = its.mesh.bsdf.pdf(&its.uv, &its.wi, &d_out_local)
                        {
                            // Compute MIS weights
                            let weight_light = mis_weight(light_pdf, pdf_bsdf);
                            if self.min_depth.map_or(true, |min| depth >= min) {
                                l_i += weight_light * throughput
                                    * its.mesh.bsdf.eval(&its.uv, &its.wi, &d_out_local)
                                    * light_record.weight;
                            }
                        }
                }

            /////////////////////////////////
            // BSDF sampling
            /////////////////////////////////
            // Compute an new direction (diffuse)
            let sampled_bsdf = match its.mesh.bsdf.sample(&its.uv, &its.wi, sampler.next2d()) {
                Some(x) => x,
                None => return l_i,
            };

            // Update the throughput
            throughput *= &sampled_bsdf.weight;

            // Generate the new ray and do the intersection
            let d_out_global = its.frame.to_world(sampled_bsdf.d);
            ray = Ray::new(its.p, d_out_global);
            its = match scene.trace(&ray) {
                Some(x) => x,
                None => return l_i,
            };

            // Check that we have intersected a light or not
            if its.mesh.is_light() && its.cos_theta() > 0.0 {
                let weight_bsdf = match sampled_bsdf.pdf {
                    PDF::SolidAngle(v) => {
                        // Know the the light is intersectable so have a solid angle PDF
                        let light_pdf = scene.direct_pdf(LightSamplingPDF::new(&ray, &its));
                        mis_weight(v, light_pdf.value())
                    }
                    PDF::Discrete(_v) => 1.0,
                    _ => panic!("Unsupported type."),
                };
                if self.min_depth.map_or(true, |min| depth >= min) {
                    l_i += throughput * (&its.mesh.emission) * weight_bsdf;
                }
            }

            // Russian roulette
            let rr_pdf = throughput.channel_max().min(0.95);
            if rr_pdf < sampler.next() {
                break;
            }
            throughput /= rr_pdf;
            // Increase the depth of the current path
            depth += 1;
        }

        l_i
    }
}