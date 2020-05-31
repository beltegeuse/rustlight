use crate::integrators::*;
use crate::volume::*;
use cgmath::{InnerSpace, Point2, Point3};

struct KullaSampling {
    // See original paper
    u_hat0: f32,
    d_l: f32,
    alpha_a: f32,
    alpha_b: f32,
    // For checking bounds
    max_dist: Option<f32>,
}
impl KullaSampling {
    fn new(max_dist: Option<f32>, ray: &Ray, pos: Point3<f32>) -> KullaSampling {
          // Compute distance on the ray
          let u_hat0 = ray.d.dot(pos - ray.o);
          // Compute D vector
          let d = pos - (ray.o + ray.d * u_hat0);
          let d_l = d.magnitude();
          
          // Compute theta_a, theta_b (angles)
          let alpha_a = u_hat0.atan2(d_l);
          let alpha_b = match max_dist {
              None => std::f32::consts::FRAC_PI_2,
              Some(v) => {
                  let u_hat1 = v + u_hat0;
                  u_hat1.atan2(d_l)
              }
          };
          assert!(alpha_a <= alpha_b);
          
          KullaSampling { u_hat0, d_l, alpha_a, alpha_b, max_dist }
    }

    fn sample(&self, sample: f32) -> (f32, f32) {
        let t = self.d_l * ((1.0 - sample)*self.alpha_a + sample*self.alpha_b).tan();
        let mut t_kulla = t - self.u_hat0;
        let pdf = self.d_l / ((self.alpha_b - self.alpha_a)*(self.d_l.powi(2) + t.powi(2)));

        // These case can happens due to numerical issues
        if t_kulla < 0.0 {
            t_kulla = 0.0;
        }
        if self.max_dist.is_some() {
            if t_kulla > self.max_dist.unwrap()  {
                t_kulla = self.max_dist.unwrap();
            }
        }
        (t_kulla, pdf)
    }

    fn pdf(&self, distance: f32) -> f32 {
        let t = distance + self.u_hat0;
        self.d_l / ((self.alpha_b - self.alpha_a)*(self.d_l.powi(2) + t.powi(2)))
    }
}

/**
 * This strategy is a bit different than medium.sample as
 * this strategy always succeed to sample a point inside the medium 
 */
struct TransmittanceSampling<'a> {
    m: &'a HomogenousVolume,
    max_dist: Option<f32>,
}
impl<'a> TransmittanceSampling<'a> {
    fn sample(&self, ray: &Ray, sample: Point2<f32>) -> (f32, f32) {
        match self.max_dist {
            None => {
                // Sample the distance proportional to the transmittance
                let sampled_dist = self.m.sample(ray, sample);
                if sampled_dist.exited {
                   // Impossible
                   panic!("Touch a surface!");
                } else {
                    (sampled_dist.t, sampled_dist.pdf)
                }
            }
            Some(v) => {
                // Select the component
                let component = (sample.x * 3.0) as u8;
                let sigma_t_c = self.m.sigma_t.get(component);
                
                // Compute the distance
                let norm = 1.0 - (-sigma_t_c * v).exp();
                let t = -(1.0 - sample.y * norm ).ln() / sigma_t_c;
                
                // Compute the pdf
                let norm_c = Color::value(1.0) - (-self.m.sigma_t * v).exp();
                let pdf = ((self.m.sigma_t / norm_c) * (-self.m.sigma_t * t).exp()).avg();
                (t, pdf)
            }
        }
    }

    fn pdf(&self, ray: &Ray, distance: f32) -> f32 {
        match self.max_dist {
            None => {
                let mut ray = *ray;
                ray.tfar = distance;
                self.m.pdf(ray, false)
            }
            Some(v) => {
                let norm_c = Color::value(1.0) - (-self.m.sigma_t * v).exp();
                ((self.m.sigma_t / norm_c) * (-self.m.sigma_t * distance).exp()).avg()
            }
        }
    }
}

/// This structure store the rendering options
/// That the user have given through the command line
#[derive(PartialEq)]
pub enum IntegratorPathKullaStrategies {
    All,                // All sampling strategies (using MIS)
    KullaPosition,      // Kulla and explicit light sampling
    TransmittancePhase, // Transmittance and phase function
}
// Note that in the original paper, the authors propose two additional strategies:
// - Kulla and phase function sampling 
// - Transmittance and explicit light sampling
// These strategies could be combined to further robustness. 
// However, the two strategies implemented already cover most of the cases.

pub struct IntegratorPathKulla {
    pub strategy: IntegratorPathKullaStrategies,
}

impl Integrator for IntegratorPathKulla {
    fn compute(&mut self, sampler: &mut dyn Sampler, accel: &dyn Acceleration, scene: &Scene) -> BufferCollection {
        compute_mc(self, sampler, accel, scene)
    }
}
impl IntegratorMC for IntegratorPathKulla {
    fn compute_pixel(
        &self,
        (ix, iy): (u32, u32),
        accel: &dyn Acceleration,
        scene: &Scene,
        sampler: &mut dyn Sampler,
        emitters: &EmitterSampler,
    ) -> Color {
        let pix = Point2::new(
            ix as f32 + sampler.next(),
            iy as f32 + sampler.next(),
        );
        let ray = scene.camera.generate(pix);

        // Get the max distance (to a surface)
        // TODO: Note that we need to revisit this approach
        //  if we wanted to support multiple participating media.
        // TODO: Only working if the sensor is inside the participating media
        let max_dist = match accel.trace(&ray) {
            None => None,
            Some(its) => Some(its.dist),
        };
        
        // Helpers
        let transmittance = |m: &HomogenousVolume, dist: f32, mut ray: Ray| -> Color {
            ray.tfar = dist;
            m.transmittance(ray)
        };
        let m = scene.volume.as_ref().unwrap();
        let phase = PhaseFunction::Isotropic();
            

        // Sampling the distance with Kulla et al.'s scheme
        let kulla_contrib = | sampler: &mut dyn Sampler | -> Color {
            if self.strategy == IntegratorPathKullaStrategies::TransmittancePhase {
                return Color::zero(); // Skip this sampling strategy
            }

            // Sampling a point on the light source
            let (_emitter, sampled_pos, flux) = emitters.random_sample_emitter_position(
                sampler.next(), sampler.next(), sampler.next2d()
            );
            
            // Sampling the distance over the sensor ray using Kulla's strategy
            let kulla_sampling = KullaSampling::new(max_dist, &ray, sampled_pos.p);    
            let (t_kulla, pdf_kulla) = kulla_sampling.sample(sampler.next());
            if pdf_kulla == 0.0 {
                return Color::zero();
            }

            // Path configuration
            let p = ray.o + ray.d * t_kulla;
            let light_w = sampled_pos.p - p;
            let light_dist = light_w.magnitude();
            let light_w = light_w / light_dist;

            // Check visibility and normals
            if sampled_pos.n.dot(-light_w) <= 0.0 {
                return Color::zero();
            }
            if !accel.visible(&p, &sampled_pos.p) {
                return Color::zero();
            }

            //////////////////////////
            // MIS computation
            let pdf_kulla =  pdf_kulla * light_dist.powi(2) / sampled_pos.n.dot(-light_w);
            let w = match self.strategy {
                IntegratorPathKullaStrategies::All => {
                    let pdf_simple = (TransmittanceSampling { m, max_dist }).pdf(&ray, t_kulla) * phase.pdf(&(-ray.d), &light_w);
                    let pdf_kulla_norm = pdf_kulla * sampled_pos.pdf.value();
                    pdf_kulla_norm.powi(2) / (pdf_simple.powi(2) + pdf_kulla_norm.powi(2))
                }
                _ => 1.0,
            };

            //////////////////////////
            // Compute contribution
            let cam_trans = transmittance(m, t_kulla, ray);
            let light_trans = transmittance(m, light_dist, ray); // FIXME: Why PI factor here?
            w * flux * cam_trans * light_trans * m.sigma_s * phase.eval(&-ray.d, &light_w) / (pdf_kulla * std::f32::consts::PI)
        };

        let contrib_phase = | sampler: &mut dyn Sampler | -> Color {
            if self.strategy == IntegratorPathKullaStrategies::KullaPosition {
                return Color::zero();
            }

            let distance_sampling = TransmittanceSampling { m, max_dist };
            let (t_dist, pdf_dist) = distance_sampling.sample(&ray, sampler.next2d());
            let sample_phase = phase.sample(&-ray.d, sampler.next2d());
            let p = ray.o + ray.d * t_dist;
            let new_ray = Ray::new(p, sample_phase.d);
            let its = accel.trace(&new_ray);
            if its.is_none() {
                return Color::zero();
            }
            let its = its.unwrap();
            if its.mesh.emission.is_zero() { 
                return Color::zero(); // Not a emitter
            }
            if its.n_s.dot(-new_ray.d) < 0.0 {
                return Color::zero();
            }

            // Configuration
            let pdf_emitter = emitters.pdf(its.mesh);
            let light_w = its.p - p;
            let light_dist = light_w.magnitude();

            //////////////////////////
            // MIS computation
            let w = match self.strategy {
                IntegratorPathKullaStrategies::All => {
                    let pdf_simple = sample_phase.pdf * pdf_dist;
                    let pdf_kulla = KullaSampling::new(max_dist, &ray, its.p).pdf(t_dist);
                    let pdf_kulla = its.mesh.pdf() * pdf_emitter * pdf_kulla * light_dist.powi(2) / its.n_s.dot(-new_ray.d);
                    pdf_simple.powi(2) / ( pdf_simple.powi(2) + pdf_kulla.powi(2) )
                }
                _ => 1.0,
            };
        
            //////////////////////////
            // Compute contribution
            let cam_trans = transmittance(m, t_dist, ray);
            let light_trans = transmittance(m, light_dist, new_ray);
            w * its.mesh.emission * cam_trans * m.sigma_s * light_trans * sample_phase.weight / pdf_dist
        };

        kulla_contrib(sampler) + contrib_phase(sampler)
    }
}
