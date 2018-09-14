use bsdfs::*;
use cgmath::{InnerSpace, Vector3};
use math::Frame;
use std;

#[derive(Deserialize)]
pub struct BSDFPhong {
    pub specular: BSDFColor,
    pub exponent: f32,
}

impl BSDF for BSDFPhong {
    fn sample(
        &self,
        uv: &Option<Vector2<f32>>,
        d_in: &Vector3<f32>,
        sample: Point2<f32>,
    ) -> Option<SampledDirection> {
        let sin_alpha = (1.0 - sample.y.powf(2.0 / (self.exponent + 1.0))).sqrt();
        let cos_alpha = sample.y.powf(1.0 / (self.exponent + 1.0));
        let phi = 2.0 * std::f32::consts::PI * sample.x;
        let local_dir = Vector3::new(sin_alpha * phi.cos(), sin_alpha * phi.sin(), cos_alpha);

        let frame = Frame::new(reflect(d_in));
        let d_out = frame.to_world(local_dir);
        if d_out.z <= 0.0 {
            None
        } else {
            let pdf = self.pdf(uv, d_in, &d_out);
            if pdf.is_zero() {
                None
            } else {
                Some(SampledDirection {
                    weight: self.eval(uv, d_in, &d_out) / pdf.value(),
                    d: d_out,
                    pdf,
                })
            }
        }
    }

    fn pdf(&self, _uv: &Option<Vector2<f32>>, d_in: &Vector3<f32>, d_out: &Vector3<f32>) -> PDF {
        if d_in.z <= 0.0 {
            return PDF::SolidAngle(0.0);
        }
        if d_out.z <= 0.0 {
            PDF::SolidAngle(0.0)
        } else {
            let alpha = reflect(d_in).dot(*d_out);
            if alpha > 0.0 {
                PDF::SolidAngle(
                    alpha.powf(self.exponent) * (self.exponent + 1.0)
                        / (2.0 * std::f32::consts::PI),
                )
            } else {
                PDF::SolidAngle(0.0)
            }
        }
    }

    fn eval(&self, uv: &Option<Vector2<f32>>, d_in: &Vector3<f32>, d_out: &Vector3<f32>) -> Color {
        if d_in.z <= 0.0 {
            return Color::zero();
        }
        if d_out.z <= 0.0 {
            Color::zero()
        } else {
            let alpha = reflect(d_in).dot(*d_out);
            if alpha > 0.0 {
                self.specular.color(uv)
                    * (alpha.powf(self.exponent) * (self.exponent + 2.0)
                        / (2.0 * std::f32::consts::PI))
            } else {
                Color::zero()
            }
        }
    }

    fn is_smooth(&self) -> bool {
        false
    }
}
