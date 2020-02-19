use crate::integrators::*;
use crate::volume::*;
use cgmath::{InnerSpace, Point2, Point3};

struct KullaConfig {
    u_hat0: f32,
    d_l: f32,
    alpha_a: f32,
    alpha_b: f32,
    // For checking bounds
    max_dist: Option<f32>,
}

impl KullaConfig {
    fn new(max_dist: Option<f32>, ray: &Ray, pos: Point3<f32>) -> KullaConfig {
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
          
          KullaConfig { u_hat0, d_l, alpha_a, alpha_b, max_dist }
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

struct DistanceSampling {
    max_dist: Option<f32>,
}

impl DistanceSampling {
    fn sample(&self, sample: f32) -> (f32, f32) {
        match self.max_dist {
            None => {
                unimplemented!();
            }
            Some(v) => {
                unimplemented!();
            }
        }
    }

    fn pdf(&self, distance: f32) -> f32 {
        match self.max_dist {
            None => {
                unimplemented!();
            }
            Some(v) => {
                unimplemented!();
            }
        }
    }
}

pub struct IntegratorPathKulla {
}

impl Integrator for IntegratorPathKulla {
    fn compute(&mut self, accel: &dyn Acceleration, scene: &Scene) -> BufferCollection {
        compute_mc(self, accel, scene)
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

        // Get the max distance
        let max_dist = match accel.trace(&ray) {
            None => None,
            Some(its) => Some(its.dist),
        };
        
        // Select an random emitter
        let (_emitter, sampled_pos, flux) = emitters.random_sample_emitter_position(sampler.next(), sampler.next(), sampler.next2d());
        let transmittance = |m: &HomogenousVolume, dist: f32, mut ray: Ray| -> Color {
            ray.tfar = dist;
            m.transmittance(ray)
        };
        let pdf_transmittance = |m: &HomogenousVolume, dist: f32, mut ray: Ray| -> f32 {
            ray.tfar = dist;
            m.pdf(ray, false)
        };
        let m = scene.volume.as_ref().unwrap();
        let phase = PhaseFunction::Isotropic();
            

        // Sampling the distance with Kulla et al.'s scheme
        let kulla_contrib = {
            let kulla_sampling = KullaConfig::new(max_dist.clone(), &ray, sampled_pos.p);    
            let (t_kulla, pdf_kulla) = kulla_sampling.sample(sampler.next());
            if pdf_kulla == 0.0 {
                return Color::zero();
            }

            // Configuration 
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

            // MIS computation
            // 4 techniques: sampling distance (kulla or transmittance) / phase function or explicit sampling
            // Might want to choose one randomly and evaluate the others (using random sampling)
            // TODO: Power heuristic works the best here
            // TODO: Might be interesting to be able to choose the different sampling strategies
            // TODO: Check how to integrate this inside path layer
            let pdf_simple = pdf_transmittance(m, t_kulla, ray.clone()) * phase.pdf(&(-ray.d), &light_w);
            let pdf_kulla = pdf_kulla * light_dist.powi(2) / sampled_pos.n.dot(-light_w); // TODO: Why not multiplying with sample_pos.pdf?
            // TODO: Check for Kulla, need to change the domain...

            // Compute contribution
            let cam_trans = transmittance(m, t_kulla, ray.clone());
            let light_trans = transmittance(m, light_dist, ray.clone());
            flux * cam_trans * light_trans * m.sigma_s * phase.eval(&-ray.d, &light_w) / (pdf_kulla * std::f32::consts::PI)
        };

        kulla_contrib
    }
}
