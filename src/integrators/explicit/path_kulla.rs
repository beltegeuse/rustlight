use crate::integrators::*;
use crate::volume::*;
use cgmath::{InnerSpace, Point2, Point3};

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
        
        let (_emitter, sampled_pos, flux) = emitters.random_sample_emitter_position(sampler.next(), sampler.next(), sampler.next2d());
        let kulla_sampling = |max_dist: Option<f32>, ray: &Ray, pos: Point3<f32>, sample: f32| {
             // Compute distance on the ray
            let u_hat0 = ray.d.dot(pos - ray.o);
            // Compute D vector
            let d = pos - (ray.o + ray.d * u_hat0);
            let d_l = d.magnitude();
            
            // Compute theta_a, theta_b
            let alpha_a = u_hat0.atan2(d_l);
            let alpha_b = match max_dist {
                None => std::f32::consts::FRAC_PI_2,
                Some(v) => {
                    let u_hat1 = v + u_hat0;
                    u_hat1.atan2(d_l)
                }
            };
            assert!(alpha_a <= alpha_b);
            let t = d_l * ((1.0 - sample)*alpha_a + sample*alpha_b).tan();
            let pdf = d_l / ((alpha_b - alpha_a)*(d_l.powi(2) + t.powi(2)));

            let mut t_kulla = t - u_hat0;
            if t_kulla < 0.0 {
                // dbg!(t_kulla);
                t_kulla = 0.0;
            }
            if max_dist.is_some() {
                if t_kulla > max_dist.unwrap()  {
                    // dbg!(max_dist, t_kulla);
                    t_kulla = max_dist.unwrap();
                }
            }
            (t_kulla, pdf)
        };
        
        let (t_kulla, pdf_kulla) = kulla_sampling(max_dist.clone(), &ray, sampled_pos.p, sampler.next());
        if pdf_kulla == 0.0 {
            return Color::zero();
        }

        let transmittance = |m: &HomogenousVolume, dist: f32, mut ray: Ray| {
            ray.tfar = dist;
            m.transmittance(ray)
        };

        // Configuration 
        let p = ray.o + ray.d * t_kulla;
        let light_w = sampled_pos.p - p;
        let light_dist = light_w.magnitude();
        let light_w = light_w / light_dist;
        if sampled_pos.n.dot(-light_w) < 0.0 {
            return Color::zero();
        }
        if !accel.visible(&p, &sampled_pos.p) {
            return Color::zero();
        }

        let m = scene.volume.as_ref().unwrap();
        let phase = PhaseFunction::Isotropic();

        let cam_trans = transmittance(m, t_kulla, ray.clone());
        let light_trans = transmittance(m, light_dist, ray.clone());
        flux * sampled_pos.n.dot(-light_w) * cam_trans * light_trans * m.sigma_s * phase.eval(&-ray.d, &light_w) / (pdf_kulla * light_dist.powi(2) * std::f32::consts::PI)
    }
}
