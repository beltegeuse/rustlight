use std;
use cgmath::{Point2,Vector3};
use cgmath::InnerSpace;

use math::basis;
use structure::Color;
use math::cosine_sample_hemisphere;

pub struct SampledDirection {
    pub weight: Color,
    pub d: Vector3<f32>,
    pub pdf: f32,
}

pub trait BSDF {
    /// sample an random direction based on the BSDF value
    /// @d_in: the incomming direction in the local space
    /// @sample: random number 2D
    /// @return: the outgoing direction, the pdf and the bsdf value $fs(...) * | n . d_out |$
    fn sample(&self, d_in: & Vector3<f32>, sample : Point2<f32>) -> Option<SampledDirection>;
    /// eval the bsdf pdf value in solid angle
    fn pdf(&self, d_in: & Vector3<f32>, d_out: & Vector3<f32>) -> f32;
    /// eval the bsdf value : $fs(...)$
    fn eval(&self, d_in: & Vector3<f32>, d_out: & Vector3<f32>) -> Color;
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BSDFDiffuse {
    pub diffuse : Color,
}
impl BSDF for BSDFDiffuse {
    fn sample(&self, d_in: &Vector3<f32>, sample: Point2<f32>) -> Option<SampledDirection> {
        if d_in.z <= 0.0 {
            None
        } else {
            let d_out = cosine_sample_hemisphere(sample);
            Some(SampledDirection {
                weight: self.diffuse,
                d: d_out,
                pdf:  d_out.z * std::f32::consts::FRAC_1_PI,
            })
        }
    }

    fn pdf(&self, d_in: &Vector3<f32>, d_out: &Vector3<f32>) -> f32 {
        assert!(d_in.z > 0.0);
        assert!(d_out.z > 0.0);
        d_out.z * std::f32::consts::FRAC_1_PI
    }

    fn eval(&self, d_in: &Vector3<f32>, d_out: &Vector3<f32>) -> Color {
        assert!(d_in.z > 0.0);
        if d_out.z > 0.0 {
            self.diffuse * d_out.z * std::f32::consts::FRAC_1_PI
        } else {
            Color::zero()
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BSDFPhong {
    pub specular: Color,
    pub exponent: f32,
}
impl BSDF for BSDFPhong {
    fn sample(&self, d_in: &Vector3<f32>, sample: Point2<f32>) -> Option<SampledDirection> {
        let sin_alpha = 1.0 - sample.y.powf(2.0 / (self.exponent + 1.0));
        let cos_alpha = sample.y.powf( 1.0 / (self.exponent + 1.0));
        let phi = 2.0 * std::f32::consts::PI * sample.x;
        let local_dir = Vector3::new(sin_alpha * phi.cos(), sin_alpha * phi.sin(), cos_alpha);

        let frame = basis(BSDFPhong::reflect(d_in));
        let d_out = frame.to_world(local_dir);
        if d_out.z <= 0.0 {
            None
        } else {
            let pdf = self.pdf(d_in, &d_out);
            if pdf == 0.0 {
                None
            } else {
                Some(SampledDirection {
                    weight: self.eval(d_in, &d_out) / pdf,
                    d: d_out,
                    pdf,
                })
            }
        }
    }

    fn pdf(&self, d_in: &Vector3<f32>, d_out: &Vector3<f32>) -> f32 {
        assert!(d_in.z > 0.0);
        assert!(d_out.z > 0.0);
        let alpha = BSDFPhong::reflect(d_in).dot(*d_out);
        alpha.powf(self.exponent) * (self.exponent + 1.0) / (2.0 * std::f32::consts::PI)
    }

    fn eval(&self, d_in: &Vector3<f32>, d_out: &Vector3<f32>) -> Color {
        assert!(d_in.z > 0.0);
        assert!(d_out.z > 0.0);
        let alpha = BSDFPhong::reflect(d_in).dot(*d_out);
        self.specular * ( std::f32::consts::FRAC_1_PI * 0.5 * alpha.powf(self.exponent) )
    }
}
impl BSDFPhong {
    fn reflect(d: &Vector3<f32>) -> Vector3<f32> {
        Vector3::new(-d.x, -d.y, d.z)
    }
}
