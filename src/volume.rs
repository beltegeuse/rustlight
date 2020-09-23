use crate::math;
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
    pub fn eval(&self, _w_i: &Vector3<f32>, _w_o: &Vector3<f32>) -> Color {
        match self {
            Self::Isotropic() => Color::value(1.0 / (std::f32::consts::PI * 4.0)),
            Self::HenyeyGreenstein(ref _g) => {
                unimplemented!();
            }
        }
    }

    pub fn pdf(&self, _w_i: &Vector3<f32>, _w_o: &Vector3<f32>) -> f32 {
        match self {
            Self::Isotropic() => 1.0 / (std::f32::consts::PI * 4.0),
            Self::HenyeyGreenstein(ref _g) => {
                unimplemented!();
            }
        }
    }

    pub fn sample(&self, _d_in: &Vector3<f32>, u: Point2<f32>) -> SampledPhase {
        match self {
            Self::Isotropic() => SampledPhase {
                d: math::sample_uniform_sphere(u),
                weight: Color::one(),
                pdf: 1.0 / (std::f32::consts::PI * 4.0),
            },
            Self::HenyeyGreenstein(ref _g) => {
                unimplemented!();
            }
        }
    }
}

// Consider isotropic participating media
pub struct HomogenousVolume {
    pub sigma_a: Color,
    pub sigma_s: Color,
    pub sigma_t: Color,
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
    pub fn sample(&self, r: &Ray, u: Point2<f32>) -> SampledDistance {
        let max_t = r.tfar;
        // Select randomly one channel
        let component = (u.x * 3.0) as u8;
        let sigma_t_c = self.sigma_t.get(component);
        // Sample a distance with the selected channel
        let t = -(1.0 - u.y).ln() / sigma_t_c;
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
