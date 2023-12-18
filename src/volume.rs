use crate::structure::*;
use cgmath::*;

// Phase function
pub struct SampledPhase {
    pub d: Vector3<f32>,
    pub weight: Color,
    pub pdf: f32,
}

#[derive(Clone, Debug)]
pub enum PhaseFunction {
    Isotropic(),
    HenyeyGreenstein(f32),
}

impl PhaseFunction {
    pub fn eval(&self, w_i: &Vector3<f32>, w_o: &Vector3<f32>) -> Color {
        match self {
            Self::Isotropic() => Color::value(1.0 / (std::f32::consts::PI * 4.0)),
            Self::HenyeyGreenstein(ref g) => {
                let tmp = 1.0 + g * g + 2.0 * g * w_i.dot(*w_o);
                Color::value(
                    std::f32::consts::FRAC_1_PI * 0.25 * (1.0 - g * g) / (tmp * tmp.sqrt()),
                )
            }
        }
    }

    pub fn pdf(&self, w_i: &Vector3<f32>, w_o: &Vector3<f32>) -> f32 {
        self.eval(w_i, w_o).avg()
    }

    pub fn sample(&self, d_in: &Vector3<f32>, u: Point2<f32>) -> SampledPhase {
        match self {
            Self::Isotropic() => SampledPhase {
                d: crate::math::sample_uniform_sphere(u),
                weight: Color::one(),
                pdf: 1.0 / (std::f32::consts::PI * 4.0),
            },
            Self::HenyeyGreenstein(ref g) => {
                let cos_theta = if g.abs() < 0.000001 {
                    // Use isotropic in this case
                    1.0 - 2.0 * u.x
                } else {
                    let sqr_term = (1.0 - g * g) / (1.0 - g + 2.0 * g * u.x);
                    (1.0 + g * g - sqr_term * sqr_term) / (2.0 * g)
                };

                let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
                let (sin_phi, cos_phi) = (2.0 * std::f32::consts::PI * u.y).sin_cos();

                // We need to inverse the direction as the two direction are pointing out
                let d_in_reverse = d_in * -1.0;
                let d = crate::math::Frame::new(d_in_reverse).to_world(Vector3::new(
                    sin_theta * cos_phi,
                    sin_theta * sin_phi,
                    cos_theta,
                ));
                SampledPhase {
                    d,
                    weight: Color::one(), // Perfect IS
                    pdf: self.pdf(d_in, &d),
                }
            }
        }
    }
}

// Consider isotropic participating media
#[derive(Clone)]
pub struct HomogenousVolume {
    pub sigma_a: Color,
    pub sigma_s: Color,
    pub sigma_t: Color,
    pub phase: PhaseFunction,
}

// Take the tungsten convention
#[derive(Clone)]
pub struct SampledDistance {
    // The real distance and weight
    pub t: f32,
    pub w: Color,
    // The continued distance and weight
    pub continued_t: f32,
    pub continued_w: Color,
    // Other informations
    pub pdf: f32,     // Probability of this event
    pub exited: bool, // if a surface have been intersected
}

// TODO: Check that sigma_t is non 0
impl HomogenousVolume {
    pub fn sample(&self, r: &Ray, u: f32) -> SampledDistance {
        let max_t = r.tfar;
        // Select randomly one channel
        let component = (u * 3.0) as u8;
        let u = u * 3.0 - component as f32;
        let sigma_t_c = self.sigma_t.get(component);
        // Sample a distance with the selected channel
        let t = -(1.0 - u).ln() / sigma_t_c;
        assert!(t >= 0.0);
        let t_min = t.min(max_t); // If there is a surface
        let exited = t >= max_t;
        // The different tau depending if we treat surfaces or not
        // compute the weight that containts the ratio between the transmittance
        // and pdf
        let tau = t_min * self.sigma_t; //< Sampled transport
        let continued_tau = t * self.sigma_t; //< Sampled transport ignoring surfaces
        let mut w = (-tau).exp();
        let continued_w = (-continued_tau).exp();
        let pdf = if exited {
            // Hit the surface
            (-tau).exp().avg()
        } else {
            // Incorporating the scattering coefficient
            // inside the transmittance weight
            w *= self.sigma_s;
            (self.sigma_t * (-tau).exp()).avg()
        };
        w /= pdf;
        // This always consider the volume only (transmittance * scattering) / (pdf sample isnide media)
        let continued_w =
            (self.sigma_s * continued_w) / (self.sigma_t * (-continued_tau).exp()).avg();
        // Finish by constructing the object
        SampledDistance {
            t: t_min,
            w,
            continued_t: t,
            continued_w,
            pdf,
            exited,
        }
    }

    pub fn transmittance(&self, r: Ray) -> Color {
        // TODO: When no intersection, transmittance need to be 0
        let tau = self.sigma_t * (r.tfar);
        (-tau).exp()
    }

    pub fn pdf(&self, r: Ray, end_on_surface: bool) -> f32 {
        let tau = self.sigma_t * (r.tfar);
        if end_on_surface {
            (-tau).exp().avg()
        } else {
            (self.sigma_t * (-tau).exp()).avg()
        }
    }
}
