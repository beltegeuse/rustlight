use crate::integrators::*;
use crate::math::*;
use crate::volume::*;
use crate::{emitter::Emitter, integrators::explicit::point_normal_poly::*};
use cgmath::{InnerSpace, Point2, Point3, Vector3};

// Trait responsible to distance sampling
pub trait DistanceSampling {
    /// Return: (distance, pdf)
    fn sample(&self, sample: f32) -> (f32, f32);
    fn pdf(&self, distance: f32) -> f32;
}

#[derive(Debug)]
pub struct EquiAngularSampling {
    // See original paper
    pub delta: f32,
    pub d_l: f32,
    pub theta_a: f32,
    pub theta_b: f32,
    // For checking bounds
    pub max_dist: Option<f32>,
    pub clamped: bool,
}

impl EquiAngularSampling {
    pub fn new(max_dist: Option<f32>, ray: &Ray, pos: &Point3<f32>) -> Self {
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
        assert!(theta_a < theta_b);

        Self {
            delta,
            d_l,
            theta_a,
            theta_b,
            max_dist,
            clamped: false,
        }
    }

    pub fn new_clamping(
        // Maximum distance along the ray
        // Optional
        max_dist: Option<f32>,
        ray: &Ray,
        // Position sampled on the light
        pos: &Point3<f32>,
        // Normal direction
        n: &Vector3<f32>,
    ) -> Option<Self> {
        // Compute distance on the ray
        let delta = ray.d.dot(pos - ray.o);
        // Compute D vector
        let d_l = (pos - (ray.o + ray.d * delta)).magnitude();

        // Compute theta_a, theta_b (angles)
        let theta_a = (-delta / d_l).atan();
        let theta_b = match max_dist {
            // Need to add a small epsilon in case of infinite ray
            // because overwise we have some singularities with ArcTan and Tan
            None => std::f32::consts::FRAC_PI_2 - 0.00001,
            Some(v) => {
                let u_hat1 = v - delta;
                (u_hat1 / d_l).atan()
            }
        };

        // Do the clampling of the angle
        let (theta_a, theta_b) = {
            // Does direction and normal aligned?
            let d_dot_n = ray.d.dot(*n);
            // Does the ray start and normal aligned?
            let p_dot_n = (pos - ray.o).dot(*n);

            // The ray look away and the origin point
            // behind the point
            if d_dot_n <= 0.0 && p_dot_n >= 0.0 {
                return None; // Nothing is visible, early quite
            }

            // Mostly parallel. Do not clamp
            if d_dot_n.abs() < 0.00001 || (p_dot_n == 0.0 && d_dot_n > 0.0) {
                (theta_a, theta_b)
            } else {
                // This case is between:
                // - All is visible (easy when computing t_hit)
                // - Some part is visible depending if the starting point is behind or front of the point
                // We just need to compute the plane intersection

                let t_hit = p_dot_n / d_dot_n;
                if t_hit < 0.0 || t_hit > max_dist.unwrap_or_else(|| std::f32::MAX) {
                    // The intersection can be behind (all visible)
                    // or all the ray is visible
                    (theta_a, theta_b)
                } else {
                    let alpha_clamp = ((t_hit - delta) / d_l).atan();
                    if p_dot_n > 0.0 {
                        // The point was behind
                        (alpha_clamp, theta_b)
                    } else {
                        // The point was in front
                        (theta_a, alpha_clamp)
                    }
                }
            }
        };

        if theta_a >= theta_b {
            None
        } else {
            Some(Self {
                delta,
                d_l,
                theta_a,
                theta_b,
                max_dist,
                clamped: true,
            })
        }
    }

    fn inside_bound(&self, distance: f32) -> bool {
        if self.clamped {
            let theta = ((distance - self.delta) / self.d_l).atan();
            theta >= self.theta_a && theta <= self.theta_b
        } else {
            true
        }
    }
}

impl DistanceSampling for EquiAngularSampling {
    fn sample(&self, sample: f32) -> (f32, f32) {
        let t = self.d_l * ((1.0 - sample) * self.theta_a + sample * self.theta_b).tan();
        let mut t_equiangular = t + self.delta;

        // p(\theta) J(\theta) = 1/(range_angle) * D/(D^2 + t^2)
        let mut pdf = (self.theta_b - self.theta_a) * (self.d_l.powi(2) + t.powi(2));
        if pdf != 0.0 {
            pdf = self.d_l / pdf;
        }

        // These case can happens due to numerical issues
        if t_equiangular < 0.0 {
            t_equiangular = 0.0;
        }
        if self.max_dist.is_some() {
            if t_equiangular > self.max_dist.unwrap() {
                t_equiangular = self.max_dist.unwrap();
            }
        }
        (t_equiangular, pdf)
    }

    fn pdf(&self, distance: f32) -> f32 {
        if !self.inside_bound(distance) {
            0.0
        } else {
            let t = distance - self.delta;
            self.d_l / ((self.theta_b - self.theta_a) * (self.d_l.powi(2) + t.powi(2)))
        }
    }
}

// All inputs are [0, 1]
pub trait Wrap {
    // PDF
    fn pdf(&self, v: f32) -> f32;
    // Sampling routine
    fn cdf_inv(&self, v: f32) -> f32;
    // Sampling inverse
    fn cdf(&self, v: f32) -> f32;
}

pub struct LinearWrap {
    pub v0: f32,
    pub v1: f32,
}

impl Wrap for LinearWrap {
    fn pdf(&self, x: f32) -> f32 {
        2.0 * (self.v0 * (1.0 - x) + self.v1 * x) / (self.v0 + self.v1)
    }

    fn cdf_inv(&self, v: f32) -> f32 {
        // Becareful about the order
        let a = self.v1 - self.v0;
        let b = 2.0 * self.v0;
        let c = -(self.v0 + self.v1) * v;
        solve_quadratic_no_opt(a, b, c)
    }

    fn cdf(&self, v: f32) -> f32 {
        v * (self.v0 * (2.0 - v) + self.v1 * v) / (self.v0 + self.v1)
    }
}

pub struct BezierWrap {
    pub v0: f32,
    pub v1: f32,
    pub v2: f32,
}
impl BezierWrap {
    fn is_valid(&self) -> bool {
        (self.v0 + self.v1 + self.v2) > 10e-6
    }
}

impl Wrap for BezierWrap {
    fn pdf(&self, x: f32) -> f32 {
        if self.is_valid() {
            let c =
                (1.0 - x).powi(2) * self.v0 + 2.0 * (1.0 - x) * x * self.v1 + x.powi(2) * self.v2;
            3.0 * c / (self.v0 + self.v1 + self.v2)
        } else {
            1.0
        }
    }

    fn cdf_inv(&self, v: f32) -> f32 {
        if !self.is_valid() {
            return v;
        }
        let inv_norm = 3.0 / (self.v0 + self.v1 + self.v2);
        // Becareful about the order
        let x_3_coef = inv_norm * (self.v0 - 2.0 * self.v1 + self.v2) / 3.0;
        let a = inv_norm * (self.v1 - self.v0);
        let b = self.v0 * inv_norm;
        let c = -v;
        // x^3 + a*x^2 + b*x + c = 0
        let valid = |v: f64| v >= 0.0 && v <= 1.0;
        let s = match roots::find_roots_cubic(x_3_coef as f64, a as f64, b as f64, c as f64) {
            roots::Roots::No(_) => todo!(),
            roots::Roots::One([s0]) => s0,
            roots::Roots::Two([s0, s1]) => {
                if valid(s0) {
                    s0
                } else {
                    s1
                }
            }
            roots::Roots::Three([s0, s1, s2]) => {
                if valid(s0) {
                    s0
                } else if valid(s1) {
                    s1
                } else {
                    s2
                }
            }
            _ => todo!(),
        };
        // Safety
        s.min(1.0).max(0.0) as f32
    }

    fn cdf(&self, x: f32) -> f32 {
        if self.is_valid() {
            ((self.v0 - 2.0 * self.v1 + self.v2) * x.powi(3)
                + 3.0 * (self.v1 - self.v0) * x.powi(2)
                + 3.0 * self.v0 * x)
                / (self.v0 + self.v1 + self.v2)
        } else {
            x
        }
    }
}

struct EquiAngularWrap<T: Wrap> {
    pub wrap: T,
    pub equiangular: EquiAngularSampling,
}
impl<T: Wrap> DistanceSampling for EquiAngularWrap<T> {
    fn sample(&self, sample: f32) -> (f32, f32) {
        let theta = self.wrap.cdf_inv(sample)
            * (self.equiangular.theta_b - self.equiangular.theta_a)
            + self.equiangular.theta_a;
        if theta < self.equiangular.theta_a || theta > self.equiangular.theta_b {
            dbg!(theta, self.equiangular.theta_a, self.equiangular.theta_b);
        }

        let t = self.equiangular.d_l * theta.tan();
        let mut t_equiangular = t + self.equiangular.delta;

        // These case can happens due to numerical issues
        if t_equiangular < 0.0 {
            t_equiangular = 0.0;
        }
        if self.equiangular.max_dist.is_some() {
            if t_equiangular > self.equiangular.max_dist.unwrap() {
                t_equiangular = self.equiangular.max_dist.unwrap();
            }
        }

        // Compute PDF
        let pdf = self.pdf(t_equiangular);
        (t_equiangular, pdf)
    }

    fn pdf(&self, distance: f32) -> f32 {
        if !self.equiangular.inside_bound(distance) {
            0.0
        } else {
            let t = distance - self.equiangular.delta;
            let theta = (t / self.equiangular.d_l).atan();
            let x = (theta - self.equiangular.theta_a)
                / (self.equiangular.theta_b - self.equiangular.theta_a);
            let pdf = self.wrap.pdf(x) / (self.equiangular.theta_b - self.equiangular.theta_a);
            pdf * self.equiangular.d_l / (self.equiangular.d_l.powi(2) + t.powi(2))
        }
    }
}

struct EquiAngularMultipleWrap<T: Wrap> {
    pub wraps: Vec<T>,
    pub equiangular: EquiAngularSampling,
}
impl<T: Wrap> DistanceSampling for EquiAngularMultipleWrap<T> {
    fn sample(&self, sample: f32) -> (f32, f32) {
        let (theta, pdf) = {
            let mut pdf = 1.0;
            let mut pos = sample;
            for w in &self.wraps {
                pos = w.cdf_inv(pos);
                pdf *= w.pdf(pos);
            }
            (
                pos * (self.equiangular.theta_b - self.equiangular.theta_a)
                    + self.equiangular.theta_a,
                pdf,
            )
        };
        let pdf = pdf / (self.equiangular.theta_b - self.equiangular.theta_a);

        if theta < self.equiangular.theta_a || theta > self.equiangular.theta_b {
            dbg!(theta, self.equiangular.theta_a, self.equiangular.theta_b);
        }

        let t = self.equiangular.d_l * theta.tan();
        let mut t_equiangular = t + self.equiangular.delta;

        // These case can happens due to numerical issues
        if t_equiangular < 0.0 {
            t_equiangular = 0.0;
        }
        if self.equiangular.max_dist.is_some() {
            if t_equiangular > self.equiangular.max_dist.unwrap() {
                t_equiangular = self.equiangular.max_dist.unwrap();
            }
        }

        let jacobian = self.equiangular.d_l / (self.equiangular.d_l.powi(2) + t.powi(2));
        let pdf = pdf * jacobian;

        (t_equiangular, pdf)
    }

    fn pdf(&self, distance: f32) -> f32 {
        if !self.equiangular.inside_bound(distance) {
            0.0
        } else {
            let t = distance - self.equiangular.delta;
            let theta = (t / self.equiangular.d_l).atan();
            let mut pos = (theta - self.equiangular.theta_a)
                / (self.equiangular.theta_b - self.equiangular.theta_a);
            let mut pdf = 1.0;
            for w in self.wraps.iter().rev() {
                pdf *= w.pdf(pos);
                pos = w.cdf(pos);
            }
            let pdf = pdf / (self.equiangular.theta_b - self.equiangular.theta_a);
            pdf * self.equiangular.d_l / (self.equiangular.d_l.powi(2) + t.powi(2))
        }
    }
}

// Functions for clamping domains
pub fn clamp_angle_tr(sigma_t: f32, d_l: f32) -> f32 {
    let clamp_angle = (0.210824 - 0.15974 * d_l * sigma_t).exp();
    clamp_angle
}
pub fn clamp_angle_phase(g: f32) -> f32 {
    let clamp_angle =
        18.8217 - 93.8831 * g + 184.173 * g.powi(2) - 160.212 * g.powi(3) + 51.7683 * g.powi(4);
    clamp_angle
}

pub struct EquiAngularTaylorSampling<T: Poly> {
    pub equiangular: EquiAngularSampling,
    pub norm: f32,
    pub cdf_a: f32,
    pub prob_poly: f32,
    pub poly: T,
    pub clamp_angle: f32,
}

impl<T: Poly> EquiAngularTaylorSampling<T> {
    pub fn new(poly: T, equiangular: EquiAngularSampling, mut clamp_angle: f32) -> Option<Self> {
        let (prob_poly, norm, cdf_a) = if equiangular.theta_a > clamp_angle {
            // Make sure that the clamp angle is correctly set
            // So that the code doing the sampling is correctly set
            clamp_angle = equiangular.theta_a;
            (0.0, 0.0, 0.0)
        } else if equiangular.theta_b > clamp_angle {
            let cdf_a = poly.cdf(equiangular.theta_a);
            let cdf_clamped = poly.cdf(clamp_angle);
            let norm = cdf_clamped - cdf_a;
            if norm <= 0.0 {
                return None;
            }

            let prob_poly = {
                let pdf_clamped = poly.pdf(clamp_angle);
                let cdf_other = pdf_clamped * (equiangular.theta_b - clamp_angle);
                norm / (norm + cdf_other)
            };

            // Taylor exp and uniform
            (prob_poly, norm, cdf_a)
        } else {
            // No uniform
            let cdf_a = poly.cdf(equiangular.theta_a);
            let cdf_b = poly.cdf(equiangular.theta_b);
            clamp_angle = equiangular.theta_b;
            let norm = cdf_b - cdf_a;
            if norm <= 0.0 {
                return None;
            }

            (1.0, norm, cdf_a)
        };

        Some(EquiAngularTaylorSampling {
            equiangular,
            norm,
            cdf_a,
            prob_poly,
            poly,
            clamp_angle,
        })
    }
}

impl<T: Poly> DistanceSampling for EquiAngularTaylorSampling<T> {
    fn sample(&self, sample: f32) -> (f32, f32) {
        // Middle of the interval
        let (theta1, pdf_angular) = if sample < self.prob_poly {
            let sample = sample / self.prob_poly;

            // Call newton step
            let theta_0 = (self.clamp_angle + self.equiangular.theta_a) * 0.5;
            let f = |v: f32| -> f32 {
                let value = (self.poly.cdf(v) - self.cdf_a) / self.norm;
                value
            };
            let f_derv = |v: f32| -> f32 {
                let value = self.poly.pdf(v) / self.norm;
                value
            };
            let res = crate::math::newton_raphson_iterate(
                theta_0,
                self.equiangular.theta_a,
                self.clamp_angle,
                30,
                10,
                sample,
                f_derv,
                f,
            );

            if res.div_zero {
                warn!("Div zero in netwon solver");
                return (0.0, 0.0); // Return dummy
            }
            (res.pos, self.prob_poly * self.poly.pdf(res.pos) / self.norm)
        } else {
            // Rescale the sample
            let sample = (sample - self.prob_poly) / (1.0 - self.prob_poly);
            let range = self.equiangular.theta_b - self.clamp_angle;

            // Uniform
            let theta = range * sample + self.clamp_angle;
            let pdf = 1.0 / range;
            (theta, pdf * (1.0 - self.prob_poly))
        };

        if theta1 < self.equiangular.theta_a || theta1 > self.equiangular.theta_b {
            dbg!(theta1, self.equiangular.theta_a, self.equiangular.theta_b);
        }

        // Compute distances
        let t = self.equiangular.d_l * theta1.tan();
        let t_equiangular = t + self.equiangular.delta;

        // Compute PDF
        let jacobian = self.equiangular.d_l / (self.equiangular.d_l.powi(2) + t.powi(2));
        let pdf = pdf_angular * jacobian;

        (t_equiangular, pdf)
    }

    fn pdf(&self, _distance: f32) -> f32 {
        unimplemented!()
    }
}

pub struct EquiAngularHybridSampling<T: Poly, W: Wrap> {
    pub equiangular: EquiAngularSampling,
    pub norm: f32,
    pub cdf_a: f32,
    pub poly: T,
    pub prob_poly: f32,
    pub warp: W,
    pub clamp_angle: f32,
}

impl<T: Poly, W: Wrap> EquiAngularHybridSampling<T, W> {
    pub fn new(
        poly: T,
        equiangular: EquiAngularSampling,
        warp: W,
        mut clamp_angle: f32,
    ) -> Option<Self> {
        let (prob_poly, norm, cdf_a) = if equiangular.theta_a > clamp_angle {
            // No taylor exp
            clamp_angle = equiangular.theta_a;
            (0.0, 0.0, 0.0)
        } else if equiangular.theta_b > clamp_angle {
            let cdf_a = poly.cdf(equiangular.theta_a);
            let norm = poly.cdf(clamp_angle) - cdf_a;
            if norm <= 0.0 {
                return None;
            }

            // Compute probability to generate poly
            let prob_poly = {
                let pdf_clamped = poly.pdf(clamp_angle);
                let cdf_other = pdf_clamped * (equiangular.theta_b - clamp_angle);
                norm / (norm + cdf_other)
            };

            (prob_poly, norm, cdf_a)
        } else {
            // No uniform
            let cdf_a = poly.cdf(equiangular.theta_a);
            let cdf_b = poly.cdf(equiangular.theta_b);
            let norm = cdf_b - cdf_a;
            if norm <= 0.0 {
                return None;
            }

            (1.0, norm, cdf_a)
        };

        Some(EquiAngularHybridSampling {
            equiangular,
            norm,
            cdf_a,
            prob_poly,
            poly,
            warp,
            clamp_angle,
        })
    }
}

impl<T: Poly, W: Wrap> DistanceSampling for EquiAngularHybridSampling<T, W> {
    fn sample(&self, sample: f32) -> (f32, f32) {
        // Middle of the interval
        let (theta1, pdf_angular) = if sample < self.prob_poly {
            let sample = sample / self.prob_poly; // Rescale
            let clamped_theta_b = self.equiangular.theta_b.min(self.clamp_angle); // TODO: Change this

            let theta_0 = (clamped_theta_b + self.equiangular.theta_a) * 0.5;

            let f = |v: f32| -> f32 {
                let value = (self.poly.cdf(v) - self.cdf_a) / self.norm;
                value
            };
            let f_derv = |v: f32| -> f32 {
                let value = self.poly.pdf(v) / self.norm;
                value
            };
            let res = crate::math::newton_raphson_iterate(
                theta_0,
                self.equiangular.theta_a,
                clamped_theta_b,
                30,
                10,
                sample,
                f_derv,
                f,
            );
            if res.div_zero {
                warn!("Div zero in netwon solver");
                return (0.0, 0.0); // Return dummy
            }
            (res.pos, self.prob_poly * self.poly.pdf(res.pos) / self.norm)
        } else {
            let sample = (sample - self.prob_poly) / (1.0 - self.prob_poly); // Rescale
            let range = self.equiangular.theta_b - self.clamp_angle;

            // Uniform
            let theta = range * sample + self.clamp_angle;
            let pdf = 1.0 / range;
            (theta, pdf * (1.0 - self.prob_poly))
        };

        let (theta2, pdf_warp) = {
            let pos = (theta1 - self.equiangular.theta_a)
                / (self.equiangular.theta_b - self.equiangular.theta_a);
            let pos = self.warp.cdf_inv(pos);
            let pdf = self.warp.pdf(pos);
            (
                pos * (self.equiangular.theta_b - self.equiangular.theta_a)
                    + self.equiangular.theta_a,
                pdf,
            )
        };

        // Compute distances
        let t = self.equiangular.d_l * theta2.tan();
        let t_equiangular = t + self.equiangular.delta;

        // Compute PDF
        let jacobian = self.equiangular.d_l / (self.equiangular.d_l.powi(2) + t.powi(2));
        let pdf = pdf_angular * pdf_warp * jacobian;

        (t_equiangular, pdf)
    }

    fn pdf(&self, _distance: f32) -> f32 {
        unimplemented!()
    }
}

/**
 * Point normal implementation
 */
pub struct PointNormalSampling {
    pub equiangular: EquiAngularSampling,
    pub a: f32,
    pub b: f32,
    pub norm: f32,
}

impl PointNormalSampling {
    pub fn new(
        max_dist: Option<f32>,
        ray: &Ray,
        pos: &Point3<f32>,
        n: &Vector3<f32>,
    ) -> Option<Self> {
        let equiangular = EquiAngularSampling::new_clamping(max_dist, ray, pos, n);
        if equiangular.is_none() {
            return None;
        }
        let equiangular = equiangular.unwrap();

        if equiangular.theta_b <= equiangular.theta_a {
            warn!(
                "Wrong equiangular angle clamp: {} > {}",
                equiangular.theta_a, equiangular.theta_b
            );
            return None;
        }

        /////////////////////
        let d = ((ray.o + ray.d * equiangular.delta) - pos) / equiangular.d_l;
        let a = n.dot(d);
        let b = n.dot(ray.d);
        let norm = a * (equiangular.theta_b.sin() - equiangular.theta_a.sin())
            - b * (equiangular.theta_b.cos() - equiangular.theta_a.cos());

        let a = a / norm;
        let b = b / norm;

        // Having negative or 0 normalization is a problem
        // This case can rarely happens due to numerical instability
        if norm <= 0.0 {
            None
        } else {
            Some(PointNormalSampling {
                equiangular,
                a,
                b,
                norm,
            })
        }
    }
}

impl DistanceSampling for PointNormalSampling {
    /// (distance, pdf)
    fn sample(&self, sample: f32) -> (f32, f32) {
        let theta = {
            let a = self.a;
            let b = self.b;

            let sample =
                sample + a * self.equiangular.theta_a.sin() - b * self.equiangular.theta_a.cos();
            let v = (a * a + b * b - (sample).powi(2)).max(0.0).sqrt();
            let q = a * sample;
            let r = b * v / a.signum();
            let s = -b * sample;
            let t = v * a.abs();

            let sol1 = (q + r).atan2(s + t);
            if sol1 >= self.equiangular.theta_a && sol1 <= self.equiangular.theta_b {
                sol1
            } else {
                (q - r).atan2(s - t)
            }
        };

        let theta = crate::clamp(theta, self.equiangular.theta_a, self.equiangular.theta_b);
        let t = self.equiangular.d_l * theta.tan();
        let t_equiangular = t + self.equiangular.delta;

        // Compute PDF
        let pdf_angular = self.a * theta.cos() + self.b * theta.sin();
        let jacobian = self.equiangular.d_l / (self.equiangular.d_l.powi(2) + t.powi(2));
        let pdf = pdf_angular * jacobian;

        (t_equiangular, pdf.abs())
    }

    fn pdf(&self, distance: f32) -> f32 {
        if !self.equiangular.inside_bound(distance) {
            0.0
        } else {
            // Compute angle & PDF & Jacobian
            let t = distance - self.equiangular.delta;
            let theta = (t / self.equiangular.d_l).atan();
            let pdf_angular = self.a * theta.cos() + self.b * theta.sin();
            let jacobian = self.equiangular.d_l / (self.equiangular.d_l.powi(2) + t.powi(2));
            let pdf = pdf_angular * jacobian;

            pdf.abs()
        }
    }
}

pub struct PointNormalTaylorSampling<T: Poly, W: Wrap> {
    pub pn: PointNormalSampling,
    pub norm_poly: f32,
    pub prob_poly: f32,
    // Other domain
    pub other_a: f32,
    pub other_b: f32,
    pub other_norm: f32,
    pub poly: T,
    pub warp: Option<W>,
    pub clamp_angle: f32,
}

impl<T: Poly, W: Wrap> PointNormalTaylorSampling<T, W> {
    // PDF normalized
    fn pdf_normalized(&self, theta: f32) -> f32 {
        if theta > self.clamp_angle {
            (1.0 - self.prob_poly) * (self.other_a * theta.cos() + self.other_b * theta.sin())
        } else {
            let p = self.poly.pdf(theta);
            let cos_theta = self.pn.a * theta.cos() + self.pn.b * theta.sin();
            p * cos_theta * self.prob_poly / self.norm_poly
        }
    }

    pub fn new(
        mut pn: PointNormalSampling,
        poly: T,
        warp: Option<W>,
        mut clamp_angle: f32,
    ) -> Option<Self> {
        let (other_a, other_b, prob_poly, norm_poly, other_norm) = if pn.equiangular.theta_a
            > clamp_angle
        {
            clamp_angle = pn.equiangular.theta_a;
            // Only uses PN for the rest of the domain
            (pn.a, pn.b, 0.0, 0.0, pn.norm)
        } else if pn.equiangular.theta_b > clamp_angle {
            // New a b for the PN between [Pi/4, theta_b]
            // -- Remove normalized factor
            let a = pn.a * pn.norm;
            let b = pn.b * pn.norm;
            // -- Renormalize a and b (with the new bounds)
            let theta_b_new = pn.equiangular.theta_b;
            let theta_a_new = clamp_angle;
            let norm_new = a * (theta_b_new.sin() - theta_a_new.sin())
                - b * (theta_b_new.cos() - theta_a_new.cos());
            let a_new = a / norm_new;
            let b_new = b / norm_new;

            // Change a and b for poly [a, clampAngle]
            let norm_poly_pn = a * (clamp_angle.sin() - pn.equiangular.theta_a.sin())
                - b * (clamp_angle.cos() - pn.equiangular.theta_a.cos());
            // - Need to renormalize as the PN bounds get changed
            pn.a = a / norm_poly_pn;
            pn.b = b / norm_poly_pn;
            pn.norm = norm_poly_pn;
            if norm_poly_pn <= 0.0 {
                return None;
            }

            // Now compute poly
            let norm_poly = poly.cdf_pn(pn.a, pn.b, pn.equiangular.theta_a, clamp_angle);

            // prob_poly: (1-Tr(..)*Pn) => Yes
            let prob_poly = {
                let pdf_clamped = {
                    let p = poly.pdf(clamp_angle);
                    let cos_theta = pn.a * clamp_angle.cos() + pn.b * clamp_angle.sin();
                    cos_theta * p
                };

                let cdf_other = pdf_clamped * (pn.equiangular.theta_b - clamp_angle);
                norm_poly / (norm_poly + cdf_other)
            };

            // Taylor exp and uniform
            (a_new, b_new, prob_poly, norm_poly, norm_new)
        } else {
            // No uniform -> Just compute the normalization factor
            // the a and b get unchanged
            let norm_poly = poly.cdf_pn(pn.a, pn.b, pn.equiangular.theta_a, pn.equiangular.theta_b);
            if norm_poly <= 0.0 {
                return None;
            }
            (0.0, 0.0, 1.0, norm_poly, 0.0)
        };

        Some(Self {
            pn,
            norm_poly,
            prob_poly,
            other_a,
            other_b,
            other_norm,
            poly,
            warp,
            clamp_angle,
        })
    }
}

impl<T: Poly, W: Wrap> DistanceSampling for PointNormalTaylorSampling<T, W> {
    fn sample(&self, sample: f32) -> (f32, f32) {
        // Sample decision for which part of the integral we sample
        let use_poly = sample < self.prob_poly;
        let (theta1, pdf_angular) = if use_poly {
            let clamped_theta_b = self.pn.equiangular.theta_b.min(self.clamp_angle);
            let sample = sample / self.prob_poly; // Rescale

            // Middle of the interval
            let theta_0 = (clamped_theta_b + self.pn.equiangular.theta_a) * 0.5;

            // Do numerical inversion
            let f = |v: f32| -> f32 {
                self.poly
                    .cdf_pn(self.pn.a, self.pn.b, self.pn.equiangular.theta_a, v)
                    / self.norm_poly
            };
            let f_derv = |v: f32| -> f32 {
                let p = self.poly.pdf(v);
                let cos_theta = self.pn.a * v.cos() + self.pn.b * v.sin();
                p * cos_theta / self.norm_poly
            };
            let res = crate::math::newton_raphson_iterate(
                theta_0,
                self.pn.equiangular.theta_a,
                clamped_theta_b,
                30,
                10,
                sample,
                f_derv,
                f,
            );
            if res.div_zero {
                warn!("Div zero in netwon solver");
                return (0.0, 0.0); // Return dummy
            }

            // This is the PDF computations
            let p = self.poly.pdf(res.pos);
            let cos_theta = self.pn.a * res.pos.cos() + self.pn.b * res.pos.sin();
            (res.pos, self.prob_poly * p * cos_theta / self.norm_poly)
        } else {
            let sample = (sample - self.prob_poly) / (1.0 - self.prob_poly);

            // [theta_a_clamped, theta_b]
            let clamped_theta_a = self.clamp_angle;
            let a = self.other_a;
            let b = self.other_b;

            // Sample PN
            let sample = sample + a * clamped_theta_a.sin() - b * clamped_theta_a.cos();
            let v = (a * a + b * b - (sample).powi(2)).max(0.0).sqrt();
            let q = a * sample;
            let r = b * v / a.signum();
            let s = -b * sample;
            let t = v * a.abs();

            // !!! atan2 different order
            let sol1 = (q + r).atan2(s + t);
            let sol = if sol1 >= clamped_theta_a && sol1 <= self.pn.equiangular.theta_b {
                sol1
            } else {
                (q - r).atan2(s - t)
            };

            // Do PN sampling
            let theta = crate::clamp(
                sol,
                self.pn.equiangular.theta_a,
                self.pn.equiangular.theta_b,
            );
            let pdf_angular = self.other_a * sol.cos() + self.other_b * sol.sin();

            ///////////////////
            // Uniform
            let pdf_warp = 1.0; // No warp performed.
            (theta, (1.0 - self.prob_poly) * pdf_angular * pdf_warp)
        };
        let theta1 = crate::clamp(
            theta1,
            self.pn.equiangular.theta_a,
            self.pn.equiangular.theta_b,
        );

        // Do the wrap
        let (theta2, pdf_warp) = match &self.warp {
            Some(w) => {
                let pos = (theta1 - self.pn.equiangular.theta_a)
                    / (self.pn.equiangular.theta_b - self.pn.equiangular.theta_a);
                let pos = w.cdf_inv(pos);
                (
                    pos * (self.pn.equiangular.theta_b - self.pn.equiangular.theta_a)
                        + self.pn.equiangular.theta_a,
                    w.pdf(pos),
                )
            }
            None => (theta1, 1.0),
        };

        // Compute distances
        let t = self.pn.equiangular.d_l * theta2.tan();
        let t_equiangular = t + self.pn.equiangular.delta;

        // Compute PDF
        let jacobian = self.pn.equiangular.d_l / (self.pn.equiangular.d_l.powi(2) + t.powi(2));
        let pdf = pdf_warp * pdf_angular * jacobian;

        if pdf == 0.0 {
            dbg!(pdf, pdf_warp, pdf_angular);
        }

        (t_equiangular, pdf)
    }

    fn pdf(&self, _distance: f32) -> f32 {
        unimplemented!()
    }
}

struct PointNormalWarpSampling<W: Wrap> {
    pub pn: PointNormalSampling,
    pub warps: Vec<W>,
}
impl<W: Wrap> DistanceSampling for PointNormalWarpSampling<W> {
    fn sample(&self, sample: f32) -> (f32, f32) {
        // Do warps
        let theta1 = {
            let a = self.pn.a;
            let b = self.pn.b;

            let sample = sample + a * self.pn.equiangular.theta_a.sin()
                - b * self.pn.equiangular.theta_a.cos();
            let v = (a * a + b * b - (sample).powi(2)).max(0.0).sqrt();
            let q = a * sample;
            let r = b * v / a.signum();
            let s = -b * sample;
            let t = v * a.abs();

            // !!! atan2 different order
            // Float sol1 = ArcTan2(S + T, Q + R);
            // Float sol2 = ArcTan2(S - T, Q - R);
            let sol1 = (q + r).atan2(s + t);
            if sol1 >= self.pn.equiangular.theta_a && sol1 <= self.pn.equiangular.theta_b {
                sol1
            } else {
                (q - r).atan2(s - t)
            }
        };
        let pdf_angular = self.pn.a * theta1.cos() + self.pn.b * theta1.sin();
        let theta1 = crate::clamp(
            theta1,
            self.pn.equiangular.theta_a,
            self.pn.equiangular.theta_b,
        );

        let (theta2, pdf_warp) = {
            let mut pdf = 1.0;
            let mut pos = (theta1 - self.pn.equiangular.theta_a)
                / (self.pn.equiangular.theta_b - self.pn.equiangular.theta_a);
            for w in &self.warps {
                pos = w.cdf_inv(pos);
                pdf *= w.pdf(pos);
            }
            (
                pos * (self.pn.equiangular.theta_b - self.pn.equiangular.theta_a)
                    + self.pn.equiangular.theta_a,
                pdf,
            )
        };

        // For debug
        if theta2 < self.pn.equiangular.theta_a || theta2 > self.pn.equiangular.theta_b {
            warn!(
                "Theta2 out range: {} [{}, {}]",
                theta2, self.pn.equiangular.theta_a, self.pn.equiangular.theta_b
            );
        }
        let theta = crate::clamp(
            theta2,
            self.pn.equiangular.theta_a,
            self.pn.equiangular.theta_b,
        );

        // Compute distances
        let t = self.pn.equiangular.d_l * theta.tan();
        let t_equiangular = t + self.pn.equiangular.delta;

        // Compute PDF
        let jacobian = self.pn.equiangular.d_l / (self.pn.equiangular.d_l.powi(2) + t.powi(2));
        let pdf = pdf_warp * pdf_angular * jacobian;

        (t_equiangular, pdf)
    }

    fn pdf(&self, distance: f32) -> f32 {
        let theta = ((distance - self.pn.equiangular.delta) / self.pn.equiangular.d_l).atan();
        if theta >= self.pn.equiangular.theta_a && theta <= self.pn.equiangular.theta_b {
            // Warps
            let mut pos = (theta - self.pn.equiangular.theta_a)
                / (self.pn.equiangular.theta_b - self.pn.equiangular.theta_a);
            let mut pdf = 1.0;
            for w in self.warps.iter().rev() {
                pdf *= w.pdf(pos);
                pos = w.cdf(pos);
            }
            let theta = pos * (self.pn.equiangular.theta_b - self.pn.equiangular.theta_a)
                + self.pn.equiangular.theta_a;

            // PN and jacobian
            let t = distance - self.pn.equiangular.delta;
            let pdf_angular = self.pn.a * theta.cos() + self.pn.b * theta.sin();
            let jacobian = self.pn.equiangular.d_l / (self.pn.equiangular.d_l.powi(2) + t.powi(2));
            pdf * pdf_angular * jacobian
        } else {
            0.0
        }
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
    fn sample(&self, sample: f32) -> (f32, f32) {
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
                let component = (sample * 3.0) as u8;
                let sample = sample * 3.0 - component as f32;
                let sigma_t_c = self.m.sigma_t.get(component);

                // Compute the distance
                let norm = 1.0 - (-sigma_t_c * v).exp();
                let t = -(1.0 - sample * norm).ln() / sigma_t_c;

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
    pub struct Strategies: u16 {
        // Distance sampling
        const TR            = 0b000000001;        // Transmittance distance sampling
        const EQUIANGULAR         = 0b000000010;  // Eq distance sampling
        const EQUIANGULAR_CLAMPED = 0b000000100;  // Eq with cos clamped
        const POINT_NORMAL  = 0b000001000;        // PN sampling
        // Flavors
        const TAYLOR_PHASE  = 0b000000010000; // Phase function
        const TAYLOR_TR     = 0b000000100000; // Tr (all distance)
        const WRAP        = 0b000010000000; // Warp approach
        const BEST        = 0b000100000000; // "Best sampling technique" (Bezier and Taylor)
                                            // Should be changed depending of the scene configuration
        // Point sampling
        const EX          = 0b010000000000;  // Explicit sampling
        const PHASE       = 0b100000000000;  // Phase function sampling
    }
}

// Different warp implemented
#[derive(Debug)]
pub enum WrapStrategy {
    Linear, // Only linear warps
    Bezier, // Only bezier warps
}

pub struct IntegratorPath {
    pub strategy: Strategies,         // Sampling strategy
    pub use_mis: bool,                // If want to use MIS (experimental)
    pub warps: String,                // Which term we want apply warp
    pub warps_strategy: WrapStrategy, // Type of warp
    pub splitting: Option<f32>,       // If we use splitting with ATS
    pub use_aa: bool,                 // Disable or not AA
}

impl Integrator for IntegratorPath {
    fn compute(
        &mut self,
        sampler: &mut dyn Sampler,
        accel: &dyn Acceleration,
        scene: &Scene,
    ) -> BufferCollection {
        match scene.nb_threads {
            None => compute_mc(self, sampler, accel, scene),
            Some(v) => {
                if v == 1 {
                    compute_mc_simple(self, sampler, accel, scene)
                } else {
                    compute_mc(self, sampler, accel, scene)
                }
            }
        }
    }
}

fn fix_random_sample_emitter<'a>(
    scene: &'a Scene,
    ray: &Ray,
    max_dist: Option<f32>,
    sampler: &mut dyn Sampler,
) -> Option<(&'a dyn Emitter, SampledPosition, Color)> {
    let emitters = scene.emitters();
    match emitters.ats {
        None => {
            let (emitter, s, f) = emitters.random_sample_emitter_position(
                sampler.next(),
                sampler.next(),
                sampler.next2d(),
            );
            Some((emitter, s, f * emitter.correct_flux()))
        }
        Some(_) => {
            if let Some((emitter, s, f)) = emitters.random_sample_emitter_position_ray(
                ray,
                max_dist,
                sampler.next(),
                sampler.next2d(),
            ) {
                Some((emitter, s, f * emitter.correct_flux()))
            } else {
                None
            }
        }
    }
}

impl IntegratorPath {
    pub fn create_distance_sampling(
        &self,
        ray: &Ray,
        p: &Point3<f32>,
        n: &Vector3<f32>,
        m: &HomogenousVolume,
        max_dist: Option<f32>,
    ) -> Option<Box<dyn DistanceSampling>> {
        if self.strategy.intersects(Strategies::EQUIANGULAR)
            || self.strategy.intersects(Strategies::EQUIANGULAR_CLAMPED)
        {
            let equiangular = if self.strategy.intersects(Strategies::EQUIANGULAR_CLAMPED) {
                let res = EquiAngularSampling::new_clamping(max_dist, &ray, p, n);
                if res.is_none() {
                    return None;
                }
                res.unwrap()
            } else {
                EquiAngularSampling::new(max_dist, &ray, p)
            };

            // Helpers for evaluating terms
            // (for the warps)
            let g = match m.phase {
                PhaseFunction::Isotropic() => 0.0,
                PhaseFunction::HenyeyGreenstein(g) => g,
            };
            let phase = |v: f32| {
                let tmp = 1.0 + g * g + 2.0 * g * v.sin();
                1.0 / (tmp * tmp.sqrt())
            };
            let pn = |v: f32| {
                let d = ((ray.o + ray.d * equiangular.delta) - p) / equiangular.d_l;
                let a = n.dot(d);
                let b = n.dot(ray.d);
                a * v.cos() + b * v.sin()
            };
            let tr = |v: f32| {
                let s_t = m.sigma_t.avg();
                (-s_t * (equiangular.d_l * v.tan() + equiangular.delta + equiangular.d_l / v.cos()))
                    .exp()
            };

            if self.strategy.intersects(Strategies::TAYLOR_PHASE) {
                match m.phase {
                    PhaseFunction::Isotropic() => Some(Box::new(equiangular)),
                    PhaseFunction::HenyeyGreenstein(g) => {
                        let poly = Poly6::phase(g); //  TODO: Change order
                        let clamp_angle = clamp_angle_phase(g);
                        EquiAngularTaylorSampling::new(poly, equiangular, clamp_angle)
                            .map_or(None, |r| Some(Box::new(r)))
                    }
                }
            } else if self.strategy.intersects(Strategies::TAYLOR_TR) {
                let poly = Poly6::tr(&equiangular, m.sigma_t.avg());
                let clamp_angle = clamp_angle_tr(m.sigma_t.avg(), equiangular.d_l);
                EquiAngularTaylorSampling::new(poly, equiangular, clamp_angle)
                    .map_or(None, |r| Some(Box::new(r)))
            } else if self.strategy.intersects(Strategies::WRAP) {
                match &self.warps_strategy {
                    WrapStrategy::Linear => {
                        let warps = self
                            .warps
                            .chars()
                            .map(|c| match c {
                                'T' => LinearWrap {
                                    v0: tr(equiangular.theta_a),
                                    v1: tr(equiangular.theta_b),
                                },
                                'N' => LinearWrap {
                                    v0: pn(equiangular.theta_a),
                                    v1: pn(equiangular.theta_b),
                                },
                                'P' => LinearWrap {
                                    v0: phase(equiangular.theta_a),
                                    v1: phase(equiangular.theta_b),
                                },
                                _ => panic!(),
                            })
                            .collect::<Vec<_>>();

                        if warps.len() == 1 {
                            Some(Box::new(EquiAngularWrap {
                                wrap: warps.into_iter().nth(0).unwrap(),
                                equiangular,
                            }))
                        } else {
                            Some(Box::new(EquiAngularMultipleWrap {
                                wraps: warps,
                                equiangular,
                            }))
                        }
                    }
                    WrapStrategy::Bezier => {
                        let mid = (equiangular.theta_b + equiangular.theta_a) / 2.0;

                        let warps = self
                            .warps
                            .chars()
                            .map(|c| match c {
                                'T' => BezierWrap {
                                    v0: tr(equiangular.theta_a),
                                    v1: tr(mid),
                                    v2: tr(equiangular.theta_b),
                                },
                                'N' => BezierWrap {
                                    v0: pn(equiangular.theta_a),
                                    v1: pn(mid),
                                    v2: pn(equiangular.theta_b),
                                },
                                'P' => BezierWrap {
                                    v0: phase(equiangular.theta_a),
                                    v1: phase(mid),
                                    v2: phase(equiangular.theta_b),
                                },
                                _ => panic!(),
                            })
                            .collect::<Vec<_>>();

                        if warps.len() == 1 {
                            Some(Box::new(EquiAngularWrap {
                                wrap: warps.into_iter().nth(0).unwrap(),
                                equiangular,
                            }))
                        } else {
                            Some(Box::new(EquiAngularMultipleWrap {
                                wraps: warps,
                                equiangular,
                            }))
                        }
                    }
                }
            } else if self.strategy.intersects(Strategies::BEST) {
                // PN Bezier + TR expension
                assert!(g != 0.0);

                // This variation can be interesting also
                // let mid = (equiangular.theta_b + equiangular.theta_a) / 2.0;
                // let warp = BezierWrap {
                //     v0: phase(equiangular.theta_a),
                //     v1: phase(mid),
                //     v2: phase(equiangular.theta_b),
                // };
                // let poly = Poly6::tr(&equiangular, m.sigma_t.avg());
                // let clamp_angle = clamp_angle_tr(m.sigma_t.avg(), equiangular.d_l);
                // EquiAngularHybridSampling::new(poly, equiangular, warp, clamp_angle)
                //     .map_or(None, |r| Some(Box::new(r)))

                // TR Warp then Phase
                let mid = (equiangular.theta_b + equiangular.theta_a) / 2.0;
                let warp = BezierWrap {
                    v0: tr(equiangular.theta_a),
                    v1: tr(mid),
                    v2: tr(equiangular.theta_b),
                };
                let poly = Poly6::phase(g);
                let clamp_angle = clamp_angle_phase(g);
                EquiAngularHybridSampling::new(poly, equiangular, warp, clamp_angle)
                    .map_or(None, |r| Some(Box::new(r)))
            } else {
                Some(Box::new(equiangular))
            }
        } else if self.strategy.intersects(Strategies::POINT_NORMAL) {
            // We are in point normal
            let res = PointNormalSampling::new(max_dist, &ray, p, n);
            if res.is_none() {
                return None;
            }
            let pn = res.unwrap();

            // See the flavor
            if self.strategy.intersects(Strategies::TAYLOR_TR) {
                let poly = Poly6::tr(&pn.equiangular, m.sigma_t.avg());
                let clamp_angle = clamp_angle_tr(m.sigma_t.avg(), pn.equiangular.d_l);
                PointNormalTaylorSampling::<_, LinearWrap>::new(pn, poly, None, clamp_angle)
                    .map_or(None, |r| Some(Box::new(r)))
            } else if self.strategy.intersects(Strategies::TAYLOR_PHASE) {
                let g = match m.phase {
                    PhaseFunction::Isotropic() => 0.0,
                    PhaseFunction::HenyeyGreenstein(g) => g,
                };
                assert!(g != 0.0);
                let poly = Poly6::phase(g);
                let clamp_angle = clamp_angle_phase(g);
                PointNormalTaylorSampling::<_, LinearWrap>::new(pn, poly, None, clamp_angle)
                    .map_or(None, |r| Some(Box::new(r)))
            } else if self.strategy.intersects(Strategies::WRAP) {
                let g = match m.phase {
                    PhaseFunction::Isotropic() => 0.0,
                    PhaseFunction::HenyeyGreenstein(g) => g,
                };

                // Closures
                let phase = |v: f32| {
                    let tmp = 1.0 + g * g + 2.0 * g * v.sin();
                    1.0 / (tmp * tmp.sqrt())
                };
                let tr = |v: f32| {
                    let s_t = m.sigma_t.avg();
                    (-s_t
                        * (pn.equiangular.d_l * v.tan()
                            + pn.equiangular.delta
                            + pn.equiangular.d_l / v.cos()))
                    .exp()
                };

                match &self.warps_strategy {
                    WrapStrategy::Linear => {
                        let warps = self
                            .warps
                            .chars()
                            .map(|c| match c {
                                'T' => LinearWrap {
                                    v0: tr(pn.equiangular.theta_a),
                                    v1: tr(pn.equiangular.theta_b),
                                },
                                'P' => LinearWrap {
                                    v0: phase(pn.equiangular.theta_a),
                                    v1: phase(pn.equiangular.theta_b),
                                },
                                _ => panic!(),
                            })
                            .collect::<Vec<_>>();
                        Some(Box::new(PointNormalWarpSampling { pn, warps }))
                    }
                    WrapStrategy::Bezier => {
                        let mid = (pn.equiangular.theta_b + pn.equiangular.theta_a) / 2.0;

                        let warps = self
                            .warps
                            .chars()
                            .map(|c| match c {
                                'T' => BezierWrap {
                                    v0: tr(pn.equiangular.theta_a),
                                    v1: tr(mid),
                                    v2: tr(pn.equiangular.theta_b),
                                },
                                'P' => BezierWrap {
                                    v0: phase(pn.equiangular.theta_a),
                                    v1: phase(mid),
                                    v2: phase(pn.equiangular.theta_b),
                                },
                                _ => panic!(),
                            })
                            .collect::<Vec<_>>();

                        Some(Box::new(PointNormalWarpSampling { pn, warps }))
                    }
                }
            } else if self.strategy.intersects(Strategies::BEST) {
                let g = match m.phase {
                    PhaseFunction::Isotropic() => 0.0,
                    PhaseFunction::HenyeyGreenstein(g) => g,
                };

                // This variation can be interesting also
                // let warp = if g == 0.0 {
                //     None
                // } else {
                //     let phase = |v: f32| {
                //         let tmp = 1.0 + g * g + 2.0 * g * v.sin();
                //         1.0 / (tmp * tmp.sqrt())
                //     };
                //     let mid = (pn.equiangular.theta_a + pn.equiangular.theta_b) / 2.0;
                //     Some(BezierWrap {
                //         v0: phase(pn.equiangular.theta_a),
                //         v1: phase(mid),
                //         v2: phase(pn.equiangular.theta_b),
                //     })
                // };
                // let poly_tr = Poly6::tr(&pn.equiangular, m.sigma_t.avg());
                // let clamp_angle = clamp_angle_tr(m.sigma_t.avg(), pn.equiangular.d_l);
                // Some(Box::new(PointNormalTaylorSampling::<_, _>::new(
                //     pn,
                //     poly_tr,
                //     warp,
                //     clamp_angle,
                // )))

                // Inverse condition
                if g == 0.0 {
                    // Do the transmittance in this case.
                    let poly_tr = Poly6::tr(&pn.equiangular, m.sigma_t.avg());
                    let clamp_angle = clamp_angle_tr(m.sigma_t.avg(), pn.equiangular.d_l);
                    PointNormalTaylorSampling::<_, BezierWrap>::new(pn, poly_tr, None, clamp_angle)
                        .map_or(None, |r| Some(Box::new(r)))
                } else {
                    // Use bezier for the transmittance
                    let tr = |v: f32| {
                        let s_t = m.sigma_t.avg();
                        (-s_t
                            * (pn.equiangular.d_l * v.tan()
                                + pn.equiangular.delta
                                + pn.equiangular.d_l / v.cos()))
                        .exp()
                    };
                    let mid = (pn.equiangular.theta_a + pn.equiangular.theta_b) / 2.0;
                    let warp = Some(BezierWrap {
                        v0: tr(pn.equiangular.theta_a),
                        v1: tr(mid),
                        v2: tr(pn.equiangular.theta_b),
                    });

                    let poly_phase = Poly6::phase(g);
                    let clamp_angle = clamp_angle_phase(g);
                    PointNormalTaylorSampling::<_, BezierWrap>::new(
                        pn,
                        poly_phase,
                        warp,
                        clamp_angle,
                    )
                    .map_or(None, |r| Some(Box::new(r)))
                }
            } else {
                Some(Box::new(pn))
            }
        } else if self.strategy.intersects(Strategies::TR) {
            Some(Box::new(TransmittanceSampling {
                m: m.clone(),
                max_dist,
                ray: ray.clone(),
            }))
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
        // Helpers
        let m = scene.volume.as_ref().unwrap();
        let transmittance = |dist: f32, mut ray: Ray| -> Color {
            ray.tfar = dist;
            m.transmittance(ray)
        };
        let compute_pdf_other_clamped =
            |pos: &Point3<f32>, n: &Vector3<f32>, distance: f32| -> f32 {
                self.create_distance_sampling(ray, pos, n, m, max_dist)
                    .map_or(0.0, |s| s.pdf(distance))
            };

        // Sample position
        let res = match scene.emitters().ats {
            Some(_) => scene.emitters().random_sample_emitter_position_ray(
                ray,
                max_dist,
                sampler.next(),
                sampler.next2d(),
            ),
            None => Some(scene.emitters().random_sample_emitter_position(
                sampler.next(),
                sampler.next(),
                sampler.next2d(),
            )),
        };
        let (_, sampled_pos, flux) = res.unwrap(); // Cannot failed
        let flux = flux * std::f32::consts::FRAC_1_PI;

        // The different strategies
        let sampling_other_clamped =
            self.create_distance_sampling(&ray, &sampled_pos.p, &sampled_pos.n, m, max_dist);
        let sampling_tr = TransmittanceSampling {
            m: m.clone(),
            max_dist,
            ray: ray.clone(),
        };
        let sampling_other = EquiAngularSampling::new(max_dist, &ray, &sampled_pos.p);

        // Generate the direction from the phase function
        let sample_phase = m.phase.sample(&-ray.d, sampler.next2d());
        // Generate compute
        #[derive(Clone)]
        struct Record {
            // Geometry
            t: f32,
            tr: Color,
            p: Point3<f32>,
            pdf: f32,
        }

        // Sample the 3 distance (or 2 if it is Equiangular)
        let res_tr = {
            let (t, pdf) = sampling_tr.sample(sampler.next());
            Record {
                t,
                tr: transmittance(t, ray.clone()) / pdf,
                p: ray.o + ray.d * t,
                pdf,
            }
        };
        let res_other = {
            let (t, pdf) = sampling_other.sample(sampler.next());
            Record {
                t,
                tr: transmittance(t, ray.clone()) / pdf,
                p: ray.o + ray.d * t,
                pdf,
            }
        };
        let res_other_clamped = if let Some(sampling_other_clamped) = &sampling_other_clamped {
            let (t, pdf) = sampling_other_clamped.sample(sampler.next());
            if pdf == 0.0 {
                None
            } else {
                let tr = transmittance(t, ray.clone()) / pdf;
                Some(Record {
                    t,
                    tr,
                    p: ray.o + ray.d * t,
                    pdf,
                })
            }
        } else {
            None
        };

        // Phase function sampling (with TR)
        let contrib_phase_tr = {
            // Generate a direction from the point p
            let new_ray = Ray::new(res_tr.p, sample_phase.d);

            // Check if we intersect an emitter
            let its = accel.trace(&new_ray);
            if its.is_none() {
                Color::zero()
            } else {
                let its = its.unwrap();
                if !its.mesh.is_light() || its.n_g.dot(-new_ray.d) < 0.0 {
                    Color::zero()
                } else {
                    let t_light = (its.p - res_tr.p).magnitude();
                    let light_trans = transmittance(t_light, new_ray.clone());
                    let pdf_phase = sample_phase.pdf;

                    // Compute MIS
                    let pdf_ex = scene
                        .emitters()
                        .direct_pdf_ray(
                            its.mesh,
                            &crate::emitter::LightSamplingPDF::new(&new_ray, &its),
                            ray,
                            max_dist,
                            its.primitive_id,
                        )
                        .value();
                    let pdf_current = pdf_phase * res_tr.pdf;

                    let pdf_other = EquiAngularSampling::new(max_dist, ray, &its.p).pdf(res_tr.t);
                    let pdf_other_clamped = compute_pdf_other_clamped(&its.p, &its.n_g, res_tr.t);

                    // Compute the set of PDFs
                    let pdfs = vec![
                        pdf_current,
                        pdf_ex * res_tr.pdf,
                        pdf_phase * pdf_other,
                        pdf_ex * pdf_other_clamped,
                    ];
                    let w = pdf_current.powi(2) / pdfs.into_iter().map(|v| v.powi(2)).sum::<f32>();

                    // Compute contrib
                    let value = its.mesh.emit(&its.uv) * m.sigma_s * sample_phase.weight;
                    w * value * res_tr.tr * light_trans
                }
            }
        };
        // Phase function sampling (with other)
        let contrib_phase_other = {
            // Generate a direction from the point p
            let new_ray = Ray::new(res_other.p, sample_phase.d);

            // Check if we intersect an emitter
            let its = accel.trace(&new_ray);
            if its.is_none() {
                Color::zero()
            } else {
                let its = its.unwrap();
                if !its.mesh.is_light() || its.n_g.dot(-new_ray.d) < 0.0 {
                    Color::zero()
                } else {
                    let t_light = (its.p - res_other.p).magnitude();
                    let light_trans = transmittance(t_light, new_ray.clone());
                    let pdf_phase = sample_phase.pdf;

                    // Compute MIS
                    let pdf_ex = scene
                        .emitters()
                        .direct_pdf_ray(
                            its.mesh,
                            &crate::emitter::LightSamplingPDF::new(&new_ray, &its),
                            ray,
                            max_dist,
                            its.primitive_id,
                        )
                        .value();

                    let pdf_other =
                        EquiAngularSampling::new(max_dist, ray, &its.p).pdf(res_other.t);
                    let pdf_other_clamped =
                        compute_pdf_other_clamped(&its.p, &its.n_g, res_other.t);

                    let pdf_tr = sampling_tr.pdf(res_other.t);
                    let pdf_current = pdf_phase * pdf_other;
                    // Compute the set of PDFs
                    // 4 strategies
                    let pdfs = vec![
                        pdf_current,
                        pdf_ex * pdf_other_clamped,
                        pdf_phase * pdf_tr,
                        pdf_ex * pdf_tr,
                    ];
                    let w = pdf_current.powi(2) / pdfs.into_iter().map(|v| v.powi(2)).sum::<f32>();

                    // Compute contrib
                    let value = its.mesh.emit(&its.uv) * m.sigma_s * sample_phase.weight;
                    w * value * res_other.tr * light_trans
                }
            }
        };

        // Explicit light sampling (with Tr)
        let contrib_ex_tr = {
            let d_light = sampled_pos.p - res_tr.p;
            let t_light = d_light.magnitude();
            if t_light == 0.0 || !t_light.is_finite() {
                Color::zero()
            } else {
                let d_light = d_light / t_light;

                // Convert domains
                let pdf_ex =
                    sampled_pos.pdf.value() * t_light.powi(2) / sampled_pos.n.dot(-d_light);
                let flux = flux * sampled_pos.n.dot(-d_light) / t_light.powi(2);

                // Backface the light or not visible
                if sampled_pos.n.dot(-d_light) <= 0.0 || !accel.visible(&res_tr.p, &sampled_pos.p) {
                    Color::zero()
                } else {
                    let light_trans = transmittance(t_light, ray.clone());

                    // Compute MIS
                    let pdf_phase = m.phase.pdf(&-ray.d, &d_light);
                    let pdf_current = pdf_ex * res_tr.pdf;
                    let pdf_other =
                        EquiAngularSampling::new(max_dist, ray, &sampled_pos.p).pdf(res_tr.t);
                    let pdf_other_clamped =
                        compute_pdf_other_clamped(&sampled_pos.p, &sampled_pos.n, res_tr.t);

                    // Compute the set of PDFs
                    let pdfs = vec![
                        pdf_current,
                        pdf_phase * res_tr.pdf,
                        pdf_phase * pdf_other,
                        pdf_ex * pdf_other_clamped,
                    ];
                    let w = pdf_current.powi(2) / pdfs.into_iter().map(|v| v.powi(2)).sum::<f32>();

                    let value = flux * m.sigma_s * m.phase.eval(&-ray.d, &d_light);
                    w * value * res_tr.tr * light_trans
                }
            }
        };

        // Explicit light sampling (with other)
        let contrib_ex_other = if let Some(res_other_clamped) = res_other_clamped {
            let d_light = sampled_pos.p - res_other_clamped.p;
            let t_light = d_light.magnitude();
            if t_light == 0.0 || !t_light.is_finite() {
                Color::zero()
            } else {
                let d_light = d_light / t_light;

                // Convert domains
                let pdf_ex =
                    sampled_pos.pdf.value() * t_light.powi(2) / sampled_pos.n.dot(-d_light);
                let flux = flux * sampled_pos.n.dot(-d_light) / t_light.powi(2);

                // Backface the light or not visible
                if sampled_pos.n.dot(-d_light) <= 0.0
                    || !accel.visible(&res_other_clamped.p, &sampled_pos.p)
                {
                    Color::zero()
                } else {
                    let light_trans = transmittance(t_light, ray.clone());

                    // Compute MIS
                    let pdf_phase = m.phase.pdf(&-ray.d, &d_light);
                    let pdf_current = pdf_ex * res_other_clamped.pdf;
                    let pdf_other = EquiAngularSampling::new(max_dist, ray, &sampled_pos.p)
                        .pdf(res_other_clamped.t);
                    let pdf_tr = sampling_tr.pdf(res_other_clamped.t);

                    // Compute the set of PDFs
                    let pdfs = vec![
                        pdf_current,
                        pdf_phase * pdf_other,
                        pdf_phase * pdf_tr,
                        pdf_ex * pdf_tr,
                    ];
                    let w = pdf_current.powi(2) / pdfs.into_iter().map(|v| v.powi(2)).sum::<f32>();
                    let value = flux * m.sigma_s * m.phase.eval(&-ray.d, &d_light);
                    w * value * res_other_clamped.tr * light_trans
                }
            }
        } else {
            Color::zero()
        };

        contrib_phase_tr + contrib_phase_other + contrib_ex_tr + contrib_ex_other
    }

    fn compute_multiple_equiangular(
        &self,
        accel: &dyn Acceleration,
        scene: &Scene,
        sampler: &mut dyn Sampler,
        ray: &Ray,
        max_dist: Option<f32>,
    ) -> Color {
        // Optimized implementation where we compute everything
        // before combining the different strategies

        // Helpers
        let m = scene.volume.as_ref().unwrap();
        let transmittance = |dist: f32, mut ray: Ray| -> Color {
            ray.tfar = dist;
            m.transmittance(ray)
        };

        // Sample distance tr
        let tr_strategy = TransmittanceSampling {
            m: m.clone(),
            max_dist,
            ray: ray.clone(),
        };
        let (t_tr, pdf_tr) = tr_strategy.sample(sampler.next());
        let transmittance_tr = transmittance(t_tr, ray.clone()) / pdf_tr;
        let p_tr = ray.o + ray.d * t_tr;

        // Now compute position on the light
        let res = match scene.emitters().ats {
            Some(_) => scene.emitters().random_sample_emitter_position_ray(
                ray,
                max_dist,
                sampler.next(),
                sampler.next2d(),
            ),
            None => Some(scene.emitters().random_sample_emitter_position(
                sampler.next(),
                sampler.next(),
                sampler.next2d(),
            )),
        };

        let (t_equiangular, transmittance_equiangular, pdf_equiangular, pdf_equiangular_prime) =
            if let Some((_, sampled_pos, _)) = &res {
                let equiangular_strategy = EquiAngularSampling::new(max_dist, ray, &sampled_pos.p);
                let (t_equiangular, pdf_equiangular) = equiangular_strategy.sample(sampler.next());
                let transmittance_equiangular =
                    transmittance(t_equiangular, ray.clone()) / pdf_equiangular;
                let pdf_equiangular_prime = equiangular_strategy.pdf(t_tr);
                (
                    t_equiangular,
                    transmittance_equiangular,
                    pdf_equiangular,
                    pdf_equiangular_prime,
                )
            } else {
                (0.0, Color::zero(), 0.0, 0.0)
            };
        let p_equiangular = ray.o + ray.d * t_equiangular;

        // Other pdf for MIS
        let pdf_tr_prime = tr_strategy.pdf(t_equiangular);

        // Sample direction
        let sample_phase = m.phase.sample(&-ray.d, sampler.next2d());

        let contrib_phase_tr = {
            // Generate a direction from the point p
            let new_ray = Ray::new(p_tr, sample_phase.d);

            // Check if we intersect an emitter
            let its = accel.trace(&new_ray);
            if its.is_none() {
                Color::zero()
            } else {
                let its = its.unwrap();
                if !its.mesh.is_light() || its.n_g.dot(-new_ray.d) < 0.0 {
                    Color::zero()
                } else {
                    let t_light = (its.p - p_tr).magnitude();
                    let light_trans = transmittance(t_light, new_ray.clone());

                    // Compute MIS
                    let pdf_phase = sample_phase.pdf;
                    let pdf_ex = scene
                        .emitters()
                        .direct_pdf_ray(
                            its.mesh,
                            &crate::emitter::LightSamplingPDF::new(&new_ray, &its),
                            ray,
                            max_dist,
                            its.primitive_id,
                        )
                        .value();
                    let pdf_current = pdf_phase * pdf_tr;
                    let pdfs = vec![
                        pdf_current,
                        pdf_ex * pdf_tr,
                        pdf_phase * pdf_equiangular_prime,
                        pdf_ex * pdf_equiangular_prime,
                    ];
                    let w = pdf_current.powi(2) / pdfs.into_iter().map(|v| v.powi(2)).sum::<f32>();

                    // Compute contrib
                    let value = its.mesh.emit(&its.uv) * m.sigma_s * sample_phase.weight;
                    w * value * transmittance_tr * light_trans
                }
            }
        };

        // Phase function sampling (with other)
        let contrib_phase_other = if res.is_some() {
            // Generate a direction from the point p
            let new_ray = Ray::new(p_equiangular, sample_phase.d);

            // Check if we intersect an emitter
            let its = accel.trace(&new_ray);
            if its.is_none() {
                Color::zero()
            } else {
                let its = its.unwrap();
                if !its.mesh.is_light() || its.n_g.dot(-new_ray.d) < 0.0 {
                    Color::zero()
                } else {
                    let t_light = (its.p - p_equiangular).magnitude();
                    let light_trans = transmittance(t_light, new_ray.clone());

                    // Compute MIS
                    let pdf_phase = sample_phase.pdf;
                    let pdf_ex = scene
                        .emitters()
                        .direct_pdf_ray(
                            its.mesh,
                            &crate::emitter::LightSamplingPDF::new(&new_ray, &its),
                            ray,
                            max_dist,
                            its.primitive_id,
                        )
                        .value();
                    let pdf_current = pdf_phase * pdf_equiangular;
                    let pdfs = vec![
                        pdf_current,
                        pdf_ex * pdf_equiangular,
                        pdf_phase * pdf_tr_prime,
                        pdf_ex * pdf_tr_prime,
                    ];
                    let w = pdf_current.powi(2) / pdfs.into_iter().map(|v| v.powi(2)).sum::<f32>();

                    // Compute contrib
                    let value = its.mesh.emit(&its.uv) * m.sigma_s * sample_phase.weight;
                    w * value * transmittance_equiangular * light_trans
                }
            }
        } else {
            Color::zero()
        };

        // Explicit light sampling (with Tr)
        let contrib_ex_tr = if let Some((_, sampled_pos, flux_pi)) = &res {
            let flux = *flux_pi * std::f32::consts::FRAC_1_PI;

            let d_light = sampled_pos.p - p_tr;
            let t_light = d_light.magnitude();
            if t_light == 0.0 || !t_light.is_finite() {
                Color::zero()
            } else {
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
                    let pdf_phase = m.phase.pdf(&-ray.d, &d_light);
                    let pdf_current = pdf_ex * pdf_tr;
                    let pdfs = vec![
                        pdf_current,
                        pdf_phase * pdf_tr,
                        pdf_phase * pdf_equiangular_prime,
                        pdf_ex * pdf_equiangular_prime,
                    ];
                    let w = pdf_current.powi(2) / pdfs.into_iter().map(|v| v.powi(2)).sum::<f32>();

                    let value = flux * m.sigma_s * m.phase.eval(&-ray.d, &d_light);
                    w * value * transmittance_tr * light_trans
                }
            }
        } else {
            Color::zero()
        };

        // Explicit light sampling (with other)
        let contrib_ex_other = if let Some((_, sampled_pos, flux_pi)) = &res {
            let flux = *flux_pi * std::f32::consts::FRAC_1_PI;

            let d_light = sampled_pos.p - p_equiangular;
            let t_light = d_light.magnitude();
            if t_light == 0.0 || !t_light.is_finite() {
                Color::zero()
            } else {
                let d_light = d_light / t_light;

                // Convert domains
                let pdf_ex =
                    sampled_pos.pdf.value() * t_light.powi(2) / sampled_pos.n.dot(-d_light);
                let flux = flux * sampled_pos.n.dot(-d_light) / t_light.powi(2);

                // Backface the light or not visible
                if sampled_pos.n.dot(-d_light) <= 0.0
                    || !accel.visible(&p_equiangular, &sampled_pos.p)
                {
                    Color::zero()
                } else {
                    let light_trans = transmittance(t_light, ray.clone());

                    // Compute MIS
                    let pdf_phase = m.phase.pdf(&-ray.d, &d_light);
                    let pdf_current = pdf_ex * pdf_equiangular;
                    let pdfs = vec![
                        pdf_current,
                        pdf_phase * pdf_equiangular,
                        pdf_phase * pdf_tr_prime,
                        pdf_ex * pdf_tr_prime,
                    ];
                    let w = pdf_current.powi(2) / pdfs.into_iter().map(|v| v.powi(2)).sum::<f32>();
                    let value = flux * m.sigma_s * m.phase.eval(&-ray.d, &d_light);
                    w * value * transmittance_equiangular * light_trans
                }
            }
        } else {
            Color::zero()
        };

        contrib_phase_tr + contrib_phase_other + contrib_ex_tr + contrib_ex_other
    }

    // MIS: Tr (Ex + Phase) with ATS Point (if enable)
    // This technique will be invalid if we use splitting
    fn compute_multiple_tr(
        &self,
        accel: &dyn Acceleration,
        scene: &Scene,
        sampler: &mut dyn Sampler,
        ray: &Ray,
        max_dist: Option<f32>,
    ) -> Color {
        assert!(self.strategy.intersects(Strategies::TR));

        // Helpers
        let m = scene.volume.as_ref().unwrap();
        let transmittance = |dist: f32, mut ray: Ray| -> Color {
            ray.tfar = dist;
            m.transmittance(ray)
        };

        // Sample distance
        let (t_cam, t_pdf) = TransmittanceSampling {
            m: m.clone(),
            max_dist,
            ray: ray.clone(),
        }
        .sample(sampler.next());

        let p = ray.o + ray.d * t_cam;
        let cam_trans = transmittance(t_cam, ray.clone()) / t_pdf;

        // Now compute position on the light
        let res = match scene.emitters().ats {
            Some(_) => scene.emitters().random_sample_emitter_position_point(
                &p,
                None,
                sampler.next(),
                sampler.next2d(),
            ),
            None => Some(scene.emitters().random_sample_emitter_position(
                sampler.next(),
                sampler.next(),
                sampler.next2d(),
            )),
        };

        let contrib_ex = if let Some((emitter, sampled_pos, flux)) = res {
            let flux = flux * emitter.correct_flux();

            let d_light = sampled_pos.p - p;
            let t_light = d_light.magnitude();
            if t_light == 0.0 || !t_light.is_finite() {
                Color::zero()
            } else {
                let d_light = d_light / t_light;

                // Convert domains
                let geom = sampled_pos.n.dot(-d_light) / t_light.powi(2);
                // Le(...) * G(...): explicit connection
                let flux = flux * geom;
                if sampled_pos.n.dot(-d_light) <= 0.0 || !accel.visible(&p, &sampled_pos.p) {
                    Color::zero()
                } else {
                    let contrib = flux * m.sigma_s * m.phase.eval(&-ray.d, &d_light);

                    // Compute MIS
                    let pdf_phase = m.phase.pdf(&-ray.d, &d_light);
                    let pdf_ex = sampled_pos.pdf.value() / geom;
                    let w = pdf_ex.powi(2) / (pdf_phase.powi(2) + pdf_ex.powi(2));

                    let light_trans = transmittance(t_light, ray.clone());
                    w * contrib * cam_trans * light_trans
                }
            }
        } else {
            Color::zero()
        };

        let sample_phase = m.phase.sample(&-ray.d, sampler.next2d());

        let contrib_phase = {
            // Generate a direction from the point p
            let new_ray = Ray::new(p, sample_phase.d);

            // Check if we intersect an emitter
            match accel.trace(&new_ray) {
                None => Color::zero(),
                Some(its) => {
                    if !its.mesh.is_light() || its.n_g.dot(-new_ray.d) < 0.0 {
                        Color::zero()
                    } else {
                        // Compute MIS
                        let pdf_phase = sample_phase.pdf;
                        let pdf_ex = scene.emitters().direct_pdf(
                            its.mesh,
                            &crate::emitter::LightSamplingPDF::new(&new_ray, &its),
                            None,
                            its.primitive_id,
                        );
                        let w = pdf_phase.powi(2) / (pdf_phase.powi(2) + pdf_ex.value().powi(2));

                        let contrib = its.mesh.emit(&its.uv) * m.sigma_s * sample_phase.weight;

                        // Compute contrib
                        let t_light = (its.p - p).magnitude();
                        let light_trans = transmittance(t_light, new_ray.clone());

                        w * contrib * cam_trans * light_trans
                    }
                }
            }
        };

        contrib_ex + contrib_phase
    }

    // Implementation of TR + EX (Optimized)
    fn compute_single_tr(
        &self,
        accel: &dyn Acceleration,
        scene: &Scene,
        sampler: &mut dyn Sampler,
        ray: &Ray,
        max_dist: Option<f32>,
    ) -> Color {
        assert!(self.strategy.intersects(Strategies::TR));
        assert!(self.strategy.intersects(Strategies::EX));

        // Helpers
        let m = scene.volume.as_ref().unwrap();
        let transmittance = |dist: f32, mut ray: Ray| -> Color {
            ray.tfar = dist;
            m.transmittance(ray)
        };

        // Sample distance
        let t_strategy = TransmittanceSampling {
            m: m.clone(),
            max_dist,
            ray: ray.clone(),
        };
        let (t_cam, t_pdf) = t_strategy.sample(sampler.next());
        let p = ray.o + ray.d * t_cam;

        // Now compute position on the light
        let res = match scene.emitters().ats {
            Some(_) => scene.emitters().random_sample_emitter_position_point(
                &p,
                None,
                sampler.next(),
                sampler.next2d(),
            ),
            None => Some(scene.emitters().random_sample_emitter_position(
                sampler.next(),
                sampler.next(),
                sampler.next2d(),
            )),
        };
        if res.is_none() {
            return Color::zero();
        }
        let (emitter, sampled_pos, flux) = res.unwrap();
        let flux = flux * emitter.correct_flux();

        let d_light = sampled_pos.p - p;
        let t_light = d_light.magnitude();
        if t_light == 0.0 || !t_light.is_finite() {
            return Color::zero();
        }
        let d_light = d_light / t_light;

        // Convert domains
        let geom = if emitter.is_surface() {
            sampled_pos.n.dot(-d_light) / t_light.powi(2)
        } else {
            1.0 / t_light.powi(2)
        };

        // Le(...) * G(...): explicit connection
        let flux = flux * geom;
        if emitter.is_surface() && sampled_pos.n.dot(-d_light) <= 0.0 {
            return Color::zero();
        }
        if !accel.visible(&p, &sampled_pos.p) {
            return Color::zero();
        }

        let contrib = flux * m.sigma_s * m.phase.eval(&-ray.d, &d_light);

        let cam_trans = transmittance(t_cam, ray.clone()) / t_pdf;
        let light_trans = transmittance(t_light, ray.clone());
        contrib * cam_trans * light_trans
    }

    // TR + Phase (Optimized)
    fn compute_single_tr_phase(
        &self,
        accel: &dyn Acceleration,
        scene: &Scene,
        sampler: &mut dyn Sampler,
        ray: &Ray,
        max_dist: Option<f32>,
    ) -> Color {
        assert!(self.strategy.intersects(Strategies::TR));
        assert!(self.strategy.intersects(Strategies::PHASE));

        // Helpers
        let m = scene.volume.as_ref().unwrap();
        let transmittance = |dist: f32, mut ray: Ray| -> Color {
            ray.tfar = dist;
            m.transmittance(ray)
        };

        // Sample distance
        let t_strategy = TransmittanceSampling {
            m: m.clone(),
            max_dist,
            ray: ray.clone(),
        };
        let (t_cam, t_pdf) = t_strategy.sample(sampler.next());
        let p = ray.o + ray.d * t_cam;

        // Now compute position on the light
        let sample_phase = m.phase.sample(&-ray.d, sampler.next2d());
        let new_ray = Ray::new(p, sample_phase.d);

        // Check if we intersect an emitter
        let its = accel.trace(&new_ray);
        if its.is_none() {
            return Color::zero();
        }
        let its = its.unwrap();
        if !its.mesh.is_light() || its.n_g.dot(-new_ray.d) < 0.0 {
            return Color::zero();
        }

        let contrib = its.mesh.emit(&its.uv) * m.sigma_s * sample_phase.weight;
        let t_light = (its.p - p).magnitude();

        let cam_trans = transmittance(t_cam, ray.clone()) / t_pdf;
        let light_trans = transmittance(t_light, ray.clone());
        contrib * cam_trans * light_trans
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
        let m = scene.volume.as_ref().unwrap();
        let transmittance = |dist: f32, mut ray: Ray| -> Color {
            ray.tfar = dist;
            m.transmittance(ray)
        };

        // Generate the distance sampling
        let (emitter, sampled_pos, flux) = {
            let res = fix_random_sample_emitter(scene, ray, max_dist, sampler);
            if res.is_none() {
                return Color::zero();
            }
            res.unwrap()
        };

        let t_strategy =
            self.create_distance_sampling(&ray, &sampled_pos.p, &sampled_pos.n, m, max_dist);

        // Generate distance
        if t_strategy.is_none() {
            // Failed
            return Color::zero();
        }
        let t_strategy = t_strategy.unwrap();
        let (t_cam, t_pdf) = t_strategy.sample(sampler.next());

        // Point
        let p = ray.o + ray.d * t_cam;
        let (contrib, t_light) = if self.strategy.intersects(Strategies::PHASE) {
            // Generate a direction from the point p
            let sample_phase = m.phase.sample(&-ray.d, sampler.next2d());
            let new_ray = Ray::new(p, sample_phase.d);

            // Check if we intersect an emitter
            let its = accel.trace(&new_ray);
            if its.is_none() {
                return Color::zero();
            }
            let its = its.unwrap();
            if !its.mesh.is_light() {
                return Color::zero(); // Not a emitter
            }
            if its.n_g.dot(-new_ray.d) < 0.0 {
                return Color::zero();
            }

            (
                its.mesh.emit(&its.uv) * m.sigma_s * sample_phase.weight,
                (its.p - p).magnitude(),
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

            let d_light = sampled_pos.p - p;
            let t_light = d_light.magnitude();
            if t_light == 0.0 || !t_light.is_finite() {
                return Color::zero();
            }
            let d_light = d_light / t_light;

            // Convert domains
            let geom = if emitter.is_surface() {
                sampled_pos.n.dot(-d_light) / t_light.powi(2)
            } else {
                1.0 / t_light.powi(2)
            };

            // pdf A -> SA
            let pdf = sampled_pos.pdf.value() / geom;
            // Le(...) * G(...): explicit connection
            let flux = flux * geom;
            let res = SampleLight {
                p: sampled_pos.p,
                n: sampled_pos.n,
                d: d_light,
                t: t_light,
                contrib: flux,
                pdf,
            };

            // Backface the light or not visible
            if emitter.is_surface() && res.n.dot(-res.d) <= 0.0 {
                return Color::zero();
            }
            if !accel.visible(&p, &res.p) {
                return Color::zero();
            }

            (
                res.contrib * m.sigma_s * m.phase.eval(&-ray.d, &res.d),
                res.t,
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

    fn compute_single_strategy_splitting(
        &self,
        accel: &dyn Acceleration,
        scene: &Scene,
        sampler: &mut dyn Sampler,
        ray: &Ray,
        max_dist: Option<f32>,
        splitting_factor: f32,
    ) -> Color {
        // Helpers
        let m = scene.volume.as_ref().unwrap();
        let transmittance = |dist: f32, mut ray: Ray| -> Color {
            ray.tfar = dist;
            m.transmittance(ray)
        };

        // Generate the distance sampling
        let mut total_contrib = Color::zero();
        for (emitter, sampled_pos, mut flux) in scene
            .emitters()
            .random_sample_emitter_position_ray_splitting(
                ray,
                max_dist,
                sampler.next(),
                sampler.next2d(),
                splitting_factor,
                sampler,
            )
        {
            // TODO: This is an non optimal fix for now
            //  Need to fix it later
            flux *= emitter.correct_flux();

            let t_strategy =
                self.create_distance_sampling(&ray, &sampled_pos.p, &sampled_pos.n, m, max_dist);

            // Generate distance
            if t_strategy.is_none() {
                // Failed
                continue;
            }
            let t_strategy = t_strategy.unwrap();
            let (t_cam, t_pdf) = t_strategy.sample(sampler.next());

            // Point
            let p = ray.o + ray.d * t_cam;
            let (contrib, t_light) = if self.strategy.intersects(Strategies::PHASE) {
                // Generate a direction from the point p
                let sample_phase = m.phase.sample(&-ray.d, sampler.next2d());
                let new_ray = Ray::new(p, sample_phase.d);

                // Check if we intersect an emitter
                let its = accel.trace(&new_ray);
                if its.is_none() {
                    continue;
                }
                let its = its.unwrap();
                if !its.mesh.is_light() {
                    continue;
                }
                if its.n_g.dot(-new_ray.d) < 0.0 {
                    continue;
                }

                (
                    its.mesh.emit(&its.uv) * m.sigma_s * sample_phase.weight,
                    (its.p - p).magnitude(),
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

                let d_light = sampled_pos.p - p;
                let t_light = d_light.magnitude();
                if t_light == 0.0 || !t_light.is_finite() {
                    continue;
                }
                let d_light = d_light / t_light;

                // Convert domains
                let geom = if emitter.is_surface() {
                    sampled_pos.n.dot(-d_light) / t_light.powi(2)
                } else {
                    1.0 / t_light.powi(2)
                };

                // pdf A -> SA
                let pdf = sampled_pos.pdf.value() / geom;
                // Le(...) * G(...): explicit connection
                let flux = flux * geom;
                let res = SampleLight {
                    p: sampled_pos.p,
                    n: sampled_pos.n,
                    d: d_light,
                    t: t_light,
                    contrib: flux,
                    pdf,
                };

                // Backface the light or not visible
                if (emitter.is_surface() && res.n.dot(-res.d) <= 0.0) || !accel.visible(&p, &res.p)
                {
                    continue;
                } else {
                    (
                        res.contrib * m.sigma_s * m.phase.eval(&-ray.d, &res.d),
                        res.t,
                    )
                }
            } else {
                unimplemented!();
            };

            //////////////////////////
            // Compute contribution
            let cam_trans = transmittance(t_cam, ray.clone()) / t_pdf;
            let light_trans = transmittance(t_light, ray.clone());
            total_contrib += contrib * cam_trans * light_trans;
        }
        total_contrib
    }
}

impl IntegratorMC for IntegratorPath {
    fn compute_pixel(
        &self,
        (ix, iy): (u32, u32),
        accel: &dyn Acceleration,
        scene: &Scene,
        sampler: &mut dyn Sampler,
    ) -> Color {
        let pix = if self.use_aa {
            Point2::new(ix as f32 + sampler.next(), iy as f32 + sampler.next())
        } else {
            Point2::new(ix as f32 + 0.5, iy as f32 + 0.5)
        };
        let ray = scene.camera.generate(pix);

        // Get the max distance (to a surface)
        let max_dist = match accel.trace(&ray) {
            None => None,
            Some(its) => Some(its.dist),
        };

        if self.use_mis {
            if self.strategy.intersects(Strategies::TR) {
                self.compute_multiple_tr(accel, scene, sampler, &ray, max_dist)
            } else if self.strategy.intersects(Strategies::EQUIANGULAR) {
                self.compute_multiple_equiangular(accel, scene, sampler, &ray, max_dist)
            } else {
                self.compute_multiple_strategy(accel, scene, sampler, &ray, max_dist)
            }
        } else {
            if self.strategy == (Strategies::TR | Strategies::EX) {
                self.compute_single_tr(accel, scene, sampler, &ray, max_dist)
            } else if self.strategy == (Strategies::TR | Strategies::PHASE) {
                self.compute_single_tr_phase(accel, scene, sampler, &ray, max_dist)
            } else {
                match self.splitting {
                    Some(splitting_factor) => self.compute_single_strategy_splitting(
                        accel,
                        scene,
                        sampler,
                        &ray,
                        max_dist,
                        splitting_factor,
                    ),
                    None => self.compute_single_strategy(accel, scene, sampler, &ray, max_dist),
                }
            }
        }
    }
}
