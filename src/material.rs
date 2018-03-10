use std;
use cgmath::{Point2,Vector3};

use structure::Color;
use math::cosine_sample_hemisphere;

pub trait BSDF {
    /// sample an random direction based on the BSDF value
    /// @d_in: the incomming direction in the local space
    /// @sample: random number 2D
    /// @return: the outgoing direction, the pdf and the bsdf value $fs(...) * | n . d_out |$
    fn sample(&self, d_in: & Vector3<f32>, sample : Point2<f32>) -> (Vector3<f32>, f32, Color);
    /// eval the bsdf pdf value in solid angle
    fn pdf(&self, d_in: & Vector3<f32>, d_out: & Vector3<f32>) -> f32;
    /// eval the bsdf value : $fs(...)$
    fn eval(&self, d_in: & Vector3<f32>, d_out: & Vector3<f32>) -> Color;
}

pub struct BSDFDiffuse {
    pub diffuse : Color,
}

impl BSDF for BSDFDiffuse {
    fn sample(&self, d_in: &Vector3<f32>, sample: Point2<f32>) -> (Vector3<f32>, f32, Color) {
        assert!(d_in.z > 0.0);
        let d_out = cosine_sample_hemisphere(sample);
        (d_out, d_out.z * std::f32::consts::FRAC_1_PI, self.diffuse.clone())
    }

    fn pdf(&self, d_in: &Vector3<f32>, d_out: &Vector3<f32>) -> f32 {
        assert!(d_in.z > 0.0);
        assert!(d_out.z > 0.0);
        d_out.z * std::f32::consts::FRAC_1_PI
    }

    fn eval(&self, d_in: &Vector3<f32>, d_out: &Vector3<f32>) -> Color {
        assert!(d_in.z > 0.0);
        if d_out.z > 0.0 {
            self.diffuse.clone() * d_out.z * std::f32::consts::FRAC_1_PI
        } else {
            Color::zero()
        }
    }
}
