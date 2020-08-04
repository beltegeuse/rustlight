use crate::bsdfs::utils::*;
use crate::bsdfs::*;
use cgmath::{InnerSpace, Point2, Vector3};

// TODO: Allow float textures for alphas
#[derive(Deserialize)]
pub struct MicrofacetDistributionBSDF {
    pub microfacet_type: MicrofacetType,
    pub alpha_u: f32,
    pub alpha_v: f32,
}

#[derive(Deserialize, Copy, Clone)]
pub enum MicrofacetType {
    Beckmann,
    GGX,
}

pub struct MicrofacetDistribution {
    pub microfacet_type: MicrofacetType,
    pub alpha_u: f32,
    pub alpha_v: f32,
}

impl MicrofacetDistribution {
    pub fn eval(&self, m: &Vector3<f32>) -> f32 {
        if cos_theta(m) <= 0.0 {
            return 0.0;
        }

        let cos_theta_2 = cos_2_theta(m);
        let beckmann_exp = ((m.x * m.x) / (self.alpha_u * self.alpha_u)
            + (m.y * m.y) / (self.alpha_v * self.alpha_v))
            / cos_theta_2;
        let res = match self.microfacet_type {
            MicrofacetType::Beckmann => {
                (-beckmann_exp).exp()
                    / (std::f32::consts::PI
                        * self.alpha_u
                        * self.alpha_v
                        * cos_theta_2
                        * cos_theta_2)
            }
            MicrofacetType::GGX => {
                let root = (1.0 + beckmann_exp) * cos_theta_2;
                1.0 / (std::f32::consts::PI * self.alpha_u * self.alpha_v * root * root)
            }
        };

        if res * cos_theta(m) < 1e-20f32 {
            0.0
        } else {
            res
        }
    }

    pub fn pdf(&self, m: &Vector3<f32>) -> f32 {
        self.eval(m) * cos_theta(m)
    }

    // (normal, pdf)
    pub fn sample(&self, sample: Point2<f32>) -> (Vector3<f32>, f32) {
        // Make sure that it is isotropic
        assert_eq!(self.alpha_u, self.alpha_v);

        let (cos_theta_m, sin_phi_m, cos_phi_m, mut pdf) = match self.microfacet_type {
            MicrofacetType::Beckmann => {
                // Isotropic code
                let (sin_phi_m, cos_phi_m) = (2.0 * std::f32::consts::PI * sample.y).sin_cos();
                let alpha_sqr = self.alpha_u * self.alpha_v;

                // Rest of the implementation
                let tan_theta_m_sqr = alpha_sqr * -(1.0f32 - sample.x).ln();
                let cos_theta_m = 1.0 / (1.0 + tan_theta_m_sqr).sqrt();
                let pdf = (1.0 - sample.x)
                    / (std::f32::consts::PI * self.alpha_u * self.alpha_v * cos_theta_m.powi(3));
                (cos_theta_m, sin_phi_m, cos_phi_m, pdf)
            }
            MicrofacetType::GGX => {
                // Isotropic code
                let (sin_phi_m, cos_phi_m) = (2.0 * std::f32::consts::PI * sample.y).sin_cos();
                let alpha_sqr = self.alpha_u * self.alpha_v;

                // Rest of the implementation
                let tan_theta_m_sqr = alpha_sqr * sample.x / (1.0 - sample.x);
                let cos_theta_m = 1.0 / (1.0 + tan_theta_m_sqr).sqrt();
                let tmp = 1.0 + tan_theta_m_sqr / alpha_sqr;
                let pdf = (std::f32::consts::FRAC_1_PI)
                    / (self.alpha_u * self.alpha_v * cos_theta_m.powi(3) * tmp.powi(2));
                (cos_theta_m, sin_phi_m, cos_phi_m, pdf)
            }
        };

        /* Prevent potential numerical issues in other stages of the model */
        if pdf < 1e-20f32 {
            pdf = 0.0;
        }

        let sin_theta_m = (1.0f32 - cos_theta_m.powi(2)).max(0.0).sqrt();
        (
            Vector3::new(
                sin_theta_m * cos_phi_m,
                sin_theta_m * sin_phi_m,
                cos_theta_m,
            ),
            pdf,
        )
    }

    pub fn g(&self, wi: &Vector3<f32>, wo: &Vector3<f32>, m: &Vector3<f32>) -> f32 {
        return self.smith_g1(wi, m) * self.smith_g1(wo, m);
    }

    pub fn smith_g1(&self, v: &Vector3<f32>, m: &Vector3<f32>) -> f32 {
        if v.dot(*m) * cos_theta(v) <= 0.0 {
            return 0.0;
        }

        /* Perpendicular incidence -- no shadowing/masking */
        let tan_theta = tan_theta(v).abs();
        if tan_theta == 0.0 {
            return 1.0;
        }

        assert_eq!(self.alpha_u, self.alpha_v);
        let alpha = self.alpha_u;
        match self.microfacet_type {
            MicrofacetType::Beckmann => {
                let a = 1.0 / (alpha * tan_theta);
                if a >= 1.6 {
                    1.0
                } else {
                    /* Use a fast and accurate (<0.35% rel. error) rational
                    approximation to the shadowing-masking function */
                    let a_sqr = a.powi(2);
                    (3.535 * a + 2.181 * a_sqr) / (1.0 + 2.276 * a + 2.577 * a_sqr)
                }
            }
            MicrofacetType::GGX => {
                let root = alpha * tan_theta;
                2.0 / (1.0 + hypot2(1.0, root))
            }
        }
    }
}
