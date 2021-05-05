use crate::integrators::*;
use crate::volume::*;
use cgmath::{InnerSpace, Point2, Point3, Vector3};

pub trait DistanceSampling {
    /// (distance, pdf)
    fn sample(&self, sample: Point2<f32>) -> (f32, f32);
    fn pdf(&self, distance: f32) -> f32;
    fn importance(&self) -> f32;
}

struct KullaSampling {
    // See original paper
    delta: f32,
    d_l: f32,
    theta_a: f32,
    theta_b: f32,
    // For checking bounds
    max_dist: Option<f32>,
}

impl KullaSampling {
    fn new(max_dist: Option<f32>, ray: &Ray, pos: Point3<f32>) -> Self {
        // Compute distance on the ray
        let delta = ray.d.dot(pos - ray.o);
        // Compute D vector
        let d_l = (pos - (ray.o + ray.d * delta)).magnitude();

        // Compute theta_a, theta_b (angles)
        let theta_a = (-delta / d_l).atan();
        let theta_b = match max_dist {
            None => std::f32::consts::FRAC_PI_2 - 0.00001,
            Some(v) => {
                let u_hat1 = v - delta;
                (u_hat1 / d_l).atan()
            }
        };
        assert!(theta_a <= theta_b);

        Self {
            delta,
            d_l,
            theta_a,
            theta_b,
            max_dist,
        }
    }
}

impl DistanceSampling for KullaSampling {
    fn sample(&self, sample: Point2<f32>) -> (f32, f32) {
        let t = self.d_l * ((1.0 - sample.x) * self.theta_a + sample.x * self.theta_b).tan();
        let mut t_kulla = t + self.delta;
        let pdf = self.d_l / ((self.theta_b - self.theta_a) * (self.d_l.powi(2) + t.powi(2)));

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
        let t = distance - self.delta;
        self.d_l / ((self.theta_b - self.theta_a) * (self.d_l.powi(2) + t.powi(2)))
    }

    fn importance(&self) -> f32 {
        (self.theta_b - self.theta_a) / self.d_l
    }
}
/**
 * This strategy is a bit different than medium.sample as
 * this strategy always succeed to sample a point inside the medium
 */
struct TransmittanceSampling {
    m: HomogenousVolume,
    max_dist: Option<f32>,
    ray: Ray,
}
impl<'a> DistanceSampling for TransmittanceSampling {
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

    fn importance(&self) -> f32 {
        1.0
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
        const PHASE         = 0b00010000;  // Phase function sampling
        const EX            = 0b00100000;  // Explicit sampling
    }
}

pub struct IntegratorPathKulla {
    pub strategy: Strategies,
    pub use_cdf: Option<usize>,
    pub use_mis: bool,
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

impl IntegratorPathKulla {
    pub fn create_distance_sampling(
        &self,
        ray: &Ray,
        scene: &Scene,
        sampler: &mut dyn Sampler,
        m: &HomogenousVolume,
        max_dist: Option<f32>,
        strategy: Strategies,
    ) -> Option<(Box<dyn DistanceSampling>, Option<(SampledPosition, Color)>)> {
        if strategy.intersects(Strategies::KULLA) {
            // Sampling a point on the light source
            // This point might be reuse if we are in Kulla + Explicit sampling
            let (_, sampled_pos, flux) = scene.emitters().random_sample_emitter_position(
                sampler.next(),
                sampler.next(),
                sampler.next2d(),
            );

            let sampling = Box::new(KullaSampling::new(max_dist, &ray, sampled_pos.p));
            Some((sampling, Some((sampled_pos, flux))))
        } else if strategy.intersects(Strategies::TR) {
            Some((
                Box::new(TransmittanceSampling {
                    m: m.clone(),
                    max_dist,
                    ray: ray.clone(),
                }),
                None,
            ))
        } else {
            todo!()
        }
    }

    fn compute_multiple_strategy(
        &self,
        accel: &dyn Acceleration,
        scene: &Scene,
        sampler: &mut dyn Sampler,
        ray: &Ray,
        max_dist: Option<f32>,
    ) -> Color {
        assert!(self.use_cdf.is_none());

        // Helpers
        let phase = PhaseFunction::Isotropic();
        let m = scene.volume.as_ref().unwrap();
        let transmittance = |dist: f32, mut ray: Ray| -> Color {
            ray.tfar = dist;
            m.transmittance(ray)
        };

        // sampling_distances: Have the different distance sampling strategies
        // flux and sampled_pos: The explicit sampling information
        let res = self.create_distance_sampling(&ray, scene, sampler, m, max_dist, self.strategy);
        let (sampling_distances, sampled_pos, flux) = if self.strategy.intersects(Strategies::TR) {
            let (_, sampled_pos, flux) = scene.emitters().random_sample_emitter_position(
                sampler.next(),
                sampler.next(),
                sampler.next2d(),
            );
            (vec![res.unwrap().0], sampled_pos, flux)
        } else {
            // We will combine with Tr
            let distance_tr: Box<dyn DistanceSampling> = Box::new(TransmittanceSampling {
                m: m.clone(),
                max_dist,
                ray: ray.clone(),
            });

            // Here the sampling strategy can failed...
            match res {
                None => {
                    // TODO: This might be not optimal as the point have already been sampled...
                    //      Even if the sampling distance failed
                    let (_, sampled_pos, flux) = scene.emitters().random_sample_emitter_position(
                        sampler.next(),
                        sampler.next(),
                        sampler.next2d(),
                    );
                    (vec![distance_tr], sampled_pos, flux)
                }
                Some(v) => {
                    let res_pos = v.1.unwrap();
                    (vec![v.0, distance_tr], res_pos.0, res_pos.1)
                }
            }
        };

        // Generate the direction from the phase function
        let sample_phase = phase.sample(&-ray.d, sampler.next2d());

        // Now we need to build 4 paths or 2 paths
        // This depends of the sampling straregy
        match &sampling_distances[..] {
            [] => Color::zero(),
            [sampling_tr] => {
                // Only one distance strartegy (Tr x [Ex, Tr])
                let (t_cam, t_pdf) = sampling_tr.sample(sampler.next2d());
                let p = ray.o + ray.d * t_cam;
                let cam_trans = transmittance(t_cam, ray.clone()) / t_pdf;

                // Phase function sampling
                let contrib_phase = {
                    // Generate a direction from the point p
                    let new_ray = Ray::new(p, sample_phase.d);

                    // Check if we intersect an emitter
                    let its = accel.trace(&new_ray);
                    if its.is_none() {
                        Color::zero()
                    } else {
                        let its = its.unwrap();
                        if its.mesh.emission.is_zero() || its.n_s.dot(-new_ray.d) < 0.0 {
                            Color::zero()
                        } else {
                            let t_light = (its.p - p).magnitude();
                            let light_trans = transmittance(t_light, new_ray.clone());
                            let pdf_phase = sample_phase.pdf;

                            // Compute MIS
                            let pdf_ex = scene.emitters().direct_pdf(
                                its.mesh,
                                &crate::emitter::LightSamplingPDF::new(&new_ray, &its),
                            );
                            let w =
                                pdf_phase.powi(2) / (pdf_phase.powi(2) + pdf_ex.value().powi(2));

                            // Compute contrib
                            let value = its.mesh.emission * m.sigma_s * sample_phase.weight;
                            w * value * cam_trans * light_trans
                        }
                    }
                };

                // Explicit light sampling
                let contrib_ex = {
                    let d_light = sampled_pos.p - p;
                    let t_light = d_light.magnitude();
                    if t_light == 0.0 || !t_light.is_finite() {
                        return Color::zero();
                    }
                    let d_light = d_light / t_light;

                    // Convert domains
                    let pdf_ex =
                        sampled_pos.pdf.value() * t_light.powi(2) / sampled_pos.n.dot(-d_light);
                    let flux = flux * sampled_pos.n.dot(-d_light) / t_light.powi(2);

                    // Backface the light or not visible
                    if sampled_pos.n.dot(-d_light) <= 0.0 || !accel.visible(&p, &sampled_pos.p) {
                        Color::zero()
                    } else {
                        let light_trans = transmittance(t_light, ray.clone());

                        // Compute MIS
                        let pdf_phase = phase.pdf(&-ray.d, &d_light);
                        let w = pdf_ex.powi(2) / (pdf_phase.powi(2) + pdf_ex.powi(2));

                        let value = flux * m.sigma_s * phase.eval(&-ray.d, &d_light);
                        w * value * cam_trans * light_trans
                    }
                };

                contrib_ex + contrib_phase
            }
            [sampling_tr, sampling_other] => {
                // The distance strategy
                // - Tr
                let (t_cam_tr, t_pdf_tr) = sampling_tr.sample(sampler.next2d());
                let p_tr = ray.o + ray.d * t_cam_tr;
                let cam_trans_tr = transmittance(t_cam_tr, ray.clone()) / t_pdf_tr;
                // - Other
                let (t_cam_other, t_pdf_other) = sampling_other.sample(sampler.next2d());
                let p_other = ray.o + ray.d * t_cam_other;
                let cam_trans_other = transmittance(t_cam_other, ray.clone()) / t_pdf_other;
                // - Usefull for MIS
                let t_pdf_other_w_tr = sampling_tr.pdf(t_cam_other);
                let t_pdf_tr_w_other = sampling_other.pdf(t_cam_tr);

                // Phase function sampling (with TR)
                let contrib_phase_tr = {
                    // Generate a direction from the point p
                    let new_ray = Ray::new(p_tr, sample_phase.d);

                    // Check if we intersect an emitter
                    let its = accel.trace(&new_ray);
                    if its.is_none() {
                        Color::zero()
                    } else {
                        let its = its.unwrap();
                        if its.mesh.emission.is_zero() || its.n_s.dot(-new_ray.d) < 0.0 {
                            Color::zero()
                        } else {
                            let t_light = (its.p - p_tr).magnitude();
                            let light_trans = transmittance(t_light, new_ray.clone());
                            let pdf_phase = sample_phase.pdf;

                            // Compute MIS
                            let pdf_ex = scene
                                .emitters()
                                .direct_pdf(
                                    its.mesh,
                                    &crate::emitter::LightSamplingPDF::new(&new_ray, &its),
                                )
                                .value();

                            let pdf_current = pdf_phase + t_pdf_tr;
                            // Compute the set of PDFs
                            let pdfs = vec![
                                pdf_current,
                                pdf_ex + t_pdf_tr,
                                pdf_phase + t_pdf_tr_w_other,
                                pdf_ex + t_pdf_tr_w_other,
                            ];
                            let w = pdf_current.powi(2)
                                / pdfs.into_iter().map(|v| v.powi(2)).sum::<f32>();

                            // Compute contrib
                            let value = its.mesh.emission * m.sigma_s * sample_phase.weight;
                            w * value * cam_trans_tr * light_trans
                        }
                    }
                };

                // Phase function sampling (with other)
                let contrib_phase_other = {
                    // Generate a direction from the point p
                    let new_ray = Ray::new(p_other, sample_phase.d);

                    // Check if we intersect an emitter
                    let its = accel.trace(&new_ray);
                    if its.is_none() {
                        Color::zero()
                    } else {
                        let its = its.unwrap();
                        if its.mesh.emission.is_zero() || its.n_s.dot(-new_ray.d) < 0.0 {
                            Color::zero()
                        } else {
                            let t_light = (its.p - p_other).magnitude();
                            let light_trans = transmittance(t_light, new_ray.clone());
                            let pdf_phase = sample_phase.pdf;

                            // Compute MIS
                            let pdf_ex = scene
                                .emitters()
                                .direct_pdf(
                                    its.mesh,
                                    &crate::emitter::LightSamplingPDF::new(&new_ray, &its),
                                )
                                .value();

                            let pdf_current = pdf_phase + t_pdf_other;
                            // Compute the set of PDFs
                            // 4 strategies
                            let pdfs = vec![
                                pdf_current,
                                pdf_ex + t_pdf_other_w_tr,
                                pdf_phase + t_pdf_other_w_tr,
                                pdf_ex + t_pdf_other,
                            ];
                            let w = pdf_current.powi(2)
                                / pdfs.into_iter().map(|v| v.powi(2)).sum::<f32>();

                            // Compute contrib
                            let value = its.mesh.emission * m.sigma_s * sample_phase.weight;
                            w * value * cam_trans_other * light_trans
                        }
                    }
                };

                // Explicit light sampling (with Tr)
                let contrib_ex_tr = {
                    let d_light = sampled_pos.p - p_tr;
                    let t_light = d_light.magnitude();
                    if t_light == 0.0 || !t_light.is_finite() {
                        return Color::zero();
                    }
                    let d_light = d_light / t_light;

                    // Convert domains
                    let pdf_ex =
                        sampled_pos.pdf.value() * t_light.powi(2) / sampled_pos.n.dot(-d_light);
                    let flux = flux * sampled_pos.n.dot(-d_light) / t_light.powi(2);

                    // Backface the light or not visible
                    if sampled_pos.n.dot(-d_light) <= 0.0 || !accel.visible(&p_tr, &sampled_pos.p) {
                        Color::zero()
                    } else {
                        let light_trans = transmittance(t_light, ray.clone());

                        // Compute MIS
                        let pdf_phase = phase.pdf(&-ray.d, &d_light);
                        let pdf_current = pdf_ex + t_pdf_tr;

                        // Compute the set of PDFs
                        let pdfs = vec![
                            pdf_current,
                            pdf_phase + t_pdf_tr,
                            pdf_phase + t_pdf_tr_w_other,
                            pdf_ex + t_pdf_tr_w_other,
                        ];
                        let w =
                            pdf_current.powi(2) / pdfs.into_iter().map(|v| v.powi(2)).sum::<f32>();

                        let value = flux * m.sigma_s * phase.eval(&-ray.d, &d_light);
                        w * value * cam_trans_tr * light_trans
                    }
                };

                // Explicit light sampling (with other)
                let contrib_ex_other = {
                    let d_light = sampled_pos.p - p_other;
                    let t_light = d_light.magnitude();
                    if t_light == 0.0 || !t_light.is_finite() {
                        return Color::zero();
                    }
                    let d_light = d_light / t_light;

                    // Convert domains
                    let pdf_ex =
                        sampled_pos.pdf.value() * t_light.powi(2) / sampled_pos.n.dot(-d_light);
                    let flux = flux * sampled_pos.n.dot(-d_light) / t_light.powi(2);

                    // Backface the light or not visible
                    if sampled_pos.n.dot(-d_light) <= 0.0
                        || !accel.visible(&p_other, &sampled_pos.p)
                    {
                        Color::zero()
                    } else {
                        let light_trans = transmittance(t_light, ray.clone());

                        // Compute MIS
                        let pdf_phase = phase.pdf(&-ray.d, &d_light);
                        let pdf_current = pdf_ex + t_pdf_other;

                        // Compute the set of PDFs
                        let pdfs = vec![
                            pdf_current,
                            pdf_phase + t_pdf_other,
                            pdf_phase + t_pdf_other_w_tr,
                            pdf_ex + t_pdf_other_w_tr,
                        ];
                        let w =
                            pdf_current.powi(2) / pdfs.into_iter().map(|v| v.powi(2)).sum::<f32>();

                        let value = flux * m.sigma_s * phase.eval(&-ray.d, &d_light);
                        w * value * cam_trans_other * light_trans
                    }
                };

                contrib_phase_tr + contrib_phase_other + contrib_ex_tr + contrib_ex_other
            }
            _ => todo!(),
        }
    }

    fn compute_single_strategy(
        &self,
        accel: &dyn Acceleration,
        scene: &Scene,
        sampler: &mut dyn Sampler,
        ray: &Ray,
        max_dist: Option<f32>,
    ) -> Color {
        // Helpers
        let phase = PhaseFunction::Isotropic();
        let m = scene.volume.as_ref().unwrap();
        let transmittance = |dist: f32, mut ray: Ray| -> Color {
            ray.tfar = dist;
            m.transmittance(ray)
        };

        // Generate the distance sampling
        let (res, pdf_sel) = match self.use_cdf {
            None => (
                self.create_distance_sampling(&ray, scene, sampler, m, max_dist, self.strategy),
                1.0,
            ),
            Some(v) => {
                let strategies: Vec<(Box<dyn DistanceSampling>, Option<(SampledPosition, Color)>)> =
                    (0..v)
                        .map(|_| {
                            self.create_distance_sampling(
                                &ray,
                                scene,
                                sampler,
                                m,
                                max_dist,
                                self.strategy,
                            )
                        })
                        .filter(|e| e.is_some())
                        .map(Option::unwrap)
                        .collect::<Vec<_>>();

                if strategies.is_empty() {
                    (None, 0.0)
                } else {
                    let mut cdf = crate::math::Distribution1DConstruct::new(strategies.len());
                    for s in &strategies {
                        cdf.add(s.0.importance());
                    }
                    let cdf = cdf.normalize();
                    let id = cdf.sample_discrete(sampler.next());
                    // w: (1 / i(x)) * (1/M) * \sum(i_x)
                    let nb = strategies.len();
                    (
                        Some(strategies.into_iter().nth(id).unwrap()),
                        cdf.pdf(id) * nb as f32,
                    )
                }
            }
        };

        // Generate distance
        if res.is_none() {
            return Color::zero();
        }
        let (t_strategy, option) = res.unwrap();
        let (t_cam, t_pdf) = t_strategy.sample(sampler.next2d());

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
            }

            // Reuse or not the sampling from KULLA
            let res = if self.strategy.intersects(Strategies::KULLA) {
                let (sampled_pos, flux) = option.unwrap();
                let d_light = sampled_pos.p - p;
                let t_light = d_light.magnitude();
                if t_light == 0.0 || !t_light.is_finite() {
                    return Color::zero();
                }
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
        contrib * cam_trans * light_trans / pdf_sel
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

        if self.use_mis {
            self.compute_multiple_strategy(accel, scene, sampler, &ray, max_dist)
        } else {
            self.compute_single_strategy(accel, scene, sampler, &ray, max_dist)
        }
    }
}
