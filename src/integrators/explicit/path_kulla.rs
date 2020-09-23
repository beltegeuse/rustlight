use crate::integrators::*;
use crate::volume::*;
use cgmath::{InnerSpace, Point2, Point3, Vector3};

pub trait DistanceSampling {
    /// (distance, pdf)
    fn sample(&self, sample: Point2<f32>) -> (f32, f32);
    fn pdf(&self, distance: f32) -> f32;
}

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
    fn new(max_dist: Option<f32>, ray: &Ray, pos: Point3<f32>) -> Self {
        // Compute distance on the ray
        let u_hat0 = ray.d.dot(pos - ray.o);
        // Compute D vector
        let d_l = (pos - (ray.o + ray.d * u_hat0)).magnitude();

        // Compute theta_a, theta_b (angles)
        let alpha_a = (u_hat0 / d_l).atan();
        let alpha_b = match max_dist {
            None => std::f32::consts::FRAC_PI_2 - 0.00001,
            Some(v) => {
                let u_hat1 = v + u_hat0;
                (u_hat1 / d_l).atan()
            }
        };
        assert!(alpha_a <= alpha_b);

        Self {
            u_hat0,
            d_l,
            alpha_a,
            alpha_b,
            max_dist,
        }
    }
}

impl DistanceSampling for KullaSampling {
    fn sample(&self, sample: Point2<f32>) -> (f32, f32) {
        let t = self.d_l * ((1.0 - sample.x) * self.alpha_a + sample.x * self.alpha_b).tan();
        let mut t_kulla = t - self.u_hat0;
        let pdf = self.d_l / ((self.alpha_b - self.alpha_a) * (self.d_l.powi(2) + t.powi(2)));

        // These case can happens due to numerical issues
        if t_kulla < 0.0 {
            t_kulla = 0.0;
        }
        if self.max_dist.is_some() {
            if t_kulla > self.max_dist.unwrap() {
                t_kulla = self.max_dist.unwrap();
            }
        }
        (t_kulla, pdf)
    }

    fn pdf(&self, distance: f32) -> f32 {
        let t = distance + self.u_hat0;
        self.d_l / ((self.alpha_b - self.alpha_a) * (self.d_l.powi(2) + t.powi(2)))
    }
}

/**
 * This strategy is a bit different than medium.sample as
 * this strategy always succeed to sample a point inside the medium
 */
struct TransmittanceSampling<'a> {
    m: &'a HomogenousVolume,
    max_dist: Option<f32>,
    ray: Ray,
}
impl<'a> DistanceSampling for TransmittanceSampling<'a> {
    fn sample(&self, sample: Point2<f32>) -> (f32, f32) {
        match self.max_dist {
            None => {
                // Sample the distance proportional to the transmittance
                let sampled_dist = self.m.sample(&self.ray, sample);
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
                let t = -(1.0 - sample.y * norm).ln() / sigma_t_c;

                // Compute the pdf
                let norm_c = Color::value(1.0) - (-self.m.sigma_t * v).exp();
                let pdf = ((self.m.sigma_t / norm_c) * (-self.m.sigma_t * t).exp()).avg();
                (t, pdf)
            }
        }
    }

    fn pdf(&self, distance: f32) -> f32 {
        match self.max_dist {
            None => {
                let mut ray = self.ray.clone();
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

struct SampleRecord<'scene> {
    /// The emitter hit
    pub emitter: &'scene dyn crate::emitter::Emitter,
    /// The configuration
    pub p0: Point3<f32>,
    pub p1: Point3<f32>,
    pub p2: Point3<f32>,
    pub d: Vector3<f32>,
}

bitflags! {
    /// This structure store the rendering options
    /// That the user have given through the command line
    pub struct Strategies: u8 {
        // Distance sampling
        const TR            = 0b00000001;  // Transmittance distance sampling
        const KULLA         = 0b00000010;  // Kulla distance sampling
        // Point sampling
        const PHASE         = 0b00001000;  // Phase function sampling
        const EX            = 0b00010000;  // Explicit sampling
    }
}

pub struct IntegratorPathKulla {
    pub strategy: Strategies,
}

impl Integrator for IntegratorPathKulla {
    fn compute(
        &mut self,
        sampler: &mut dyn Sampler,
        accel: &dyn Acceleration,
        scene: &Scene,
    ) -> BufferCollection {
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
    ) -> Color {
        let pix = Point2::new(ix as f32 + sampler.next(), iy as f32 + sampler.next());
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
        let phase = PhaseFunction::Isotropic();
        let m = scene.volume.as_ref().unwrap();
        let transmittance = |dist: f32, mut ray: Ray| -> Color {
            ray.tfar = dist;
            m.transmittance(ray)
        };

        // Sampling the distance with Kulla et al.'s scheme
        let (t_cam, t_pdf, option) = if self.strategy.intersects(Strategies::KULLA) {
            // Sampling a point on the light source
            // This point might be reuse if we are in Kulla + Explicit sampling
            let (_, sampled_pos, flux) = scene.emitters().random_sample_emitter_position(
                sampler.next(),
                sampler.next(),
                sampler.next2d(),
            );

            // Sampling the distance over the sensor ray using Kulla's strategy
            let kulla_sampling = KullaSampling::new(max_dist, &ray, sampled_pos.p);

            let (t_kulla, pdf_kulla) = kulla_sampling.sample(sampler.next2d());
            if pdf_kulla == 0.0 {
                return Color::zero();
            }
            (t_kulla, pdf_kulla, Some((sampled_pos, flux)))
        } else if self.strategy.intersects(Strategies::TR) {
            // Sampling the distance
            let distance_sampling = TransmittanceSampling {
                m,
                max_dist,
                ray: ray.clone(),
            };
            let (t_tr, pdf_tr) = distance_sampling.sample(sampler.next2d());
            (t_tr, pdf_tr, None)
        } else {
            panic!("A distance sampling technique need to be provided");
        };

        // Point
        let p = ray.o + ray.d * t_cam;
        let (contrib, t_light, _pdf_light) = if self.strategy.intersects(Strategies::PHASE) {
            // Generate a direction from the point p
            let sample_phase = phase.sample(&-ray.d, sampler.next2d());
            let new_ray = Ray::new(p, sample_phase.d);

            // Check if we intersect an emitter
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

            (
                its.mesh.emission * m.sigma_s * sample_phase.weight,
                (its.p - p).magnitude(),
                sample_phase.pdf,
            )
        } else if self.strategy.intersects(Strategies::EX) {
            struct SampleLight {
                pub p: Point3<f32>,
                pub n: Vector3<f32>,
                pub d: Vector3<f32>,
                pub t: f32,
                pub contrib: Color,
                pub pdf: f32,
            };

            // Reuse or not the sampling from KULLA
            let res = if self.strategy.intersects(Strategies::KULLA) {
                let (sampled_pos, flux) = option.unwrap();
                let d_light = sampled_pos.p - p;
                let t_light = d_light.magnitude();
                let d_light = d_light / t_light;

                // Convert domains
                let pdf = sampled_pos.pdf.value() * t_light.powi(2) / sampled_pos.n.dot(-d_light);
                let flux = flux * sampled_pos.n.dot(-d_light) / t_light.powi(2);
                SampleLight {
                    p: sampled_pos.p,
                    n: sampled_pos.n,
                    d: d_light,
                    t: t_light,
                    contrib: flux,
                    pdf,
                }
            } else {
                let res = scene.emitters().sample_light(
                    &p,
                    sampler.next(),
                    sampler.next(),
                    sampler.next2d(),
                );

                SampleLight {
                    p: res.p,
                    n: res.n,
                    d: res.d,
                    t: (res.p - p).magnitude(),
                    contrib: res.weight,
                    pdf: res.pdf.value(),
                }
            };

            // Backface the light or not visible
            if res.n.dot(-res.d) <= 0.0 {
                return Color::zero();
            }
            if !accel.visible(&p, &res.p) {
                return Color::zero();
            }

            // TODO: Check if PI is needed
            (
                res.contrib * m.sigma_s * phase.eval(&-ray.d, &res.d),
                res.t,
                res.pdf,
            )
        } else {
            unimplemented!();
        };

        //////////////////////////
        // Compute contribution
        let cam_trans = transmittance(t_cam, ray.clone()) / t_pdf;
        let light_trans = transmittance(t_light, ray.clone());
        contrib * cam_trans * light_trans
    }
}
