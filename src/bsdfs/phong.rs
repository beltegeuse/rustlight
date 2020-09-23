use crate::bsdfs::*;
use crate::math::{cosine_sample_hemisphere, Frame};
use cgmath::{InnerSpace, Vector3};
use std;

pub struct BSDFPhong {
    pub diffuse: BSDFColor,
    pub specular: BSDFColor,
    pub exponent: f32,
    pub weight_specular: f32,
}

impl BSDF for BSDFPhong {
    fn sample(
        &self,
        uv: &Option<Vector2<f32>>,
        d_in: &Vector3<f32>,
        mut sample: Point2<f32>,
        transport: Transport,
    ) -> Option<SampledDirection> {
        if d_in.z <= 0.0 {
            return None;
        }

        let d_out = if sample.x < self.weight_specular {
            sample.x /= self.weight_specular;

            let sin_alpha = (1.0 - sample.y.powf(2.0 / (self.exponent + 1.0))).sqrt();
            let cos_alpha = sample.y.powf(1.0 / (self.exponent + 1.0));
            let phi = 2.0 * std::f32::consts::PI * sample.x;
            let local_dir = Vector3::new(sin_alpha * phi.cos(), sin_alpha * phi.sin(), cos_alpha);

            let frame = Frame::new(reflect(d_in));
            let d_out = frame.to_world(local_dir);
            if d_out.z <= 0.0 {
                None
            } else {
                Some(d_out)
            }
        } else {
            sample.x = (sample.x - self.weight_specular) / (1.0 - self.weight_specular);
            Some(cosine_sample_hemisphere(sample))
        };

        if d_out.is_none() {
            return None;
        }

        let d_out = d_out.unwrap();
        let pdf = self.pdf(uv, d_in, &d_out, Domain::SolidAngle, transport);
        if pdf.value() == 0.0 {
            None
        } else {
            Some(SampledDirection {
                weight: self.eval(uv, d_in, &d_out, Domain::SolidAngle, transport) / pdf.value(),
                d: d_out,
                pdf,
                eta: 1.0,
                event: BSDFEvent::REFLECTION,
                event_type: BSDFType::GLOSSY,
            })
        }
    }

    fn pdf(
        &self,
        _uv: &Option<Vector2<f32>>,
        d_in: &Vector3<f32>,
        d_out: &Vector3<f32>,
        domain: Domain,
        _: Transport,
    ) -> PDF {
        assert!(domain == Domain::SolidAngle);

        if d_in.z <= 0.0 || d_out.z <= 0.0 {
            return PDF::SolidAngle(0.0);
        }

        let pdf_specular = {
            let alpha = reflect(d_in).dot(*d_out);
            if alpha > 0.0 {
                self.weight_specular * alpha.powf(self.exponent) * (self.exponent + 1.0)
                    / (2.0 * std::f32::consts::PI)
            } else {
                0.0
            }
        };
        let pdf_diffuse = (1.0 - self.weight_specular) * d_out.z * std::f32::consts::FRAC_1_PI;

        PDF::SolidAngle(pdf_specular + pdf_diffuse)
    }

    fn eval(
        &self,
        uv: &Option<Vector2<f32>>,
        d_in: &Vector3<f32>,
        d_out: &Vector3<f32>,
        domain: Domain,
        _: Transport,
    ) -> Color {
        assert!(domain == Domain::SolidAngle);

        if d_in.z <= 0.0 || d_out.z <= 0.0 {
            return Color::zero();
        }
        let specular_value = {
            let alpha = reflect(d_in).dot(*d_out);
            if alpha > 0.0 {
                self.specular.color(uv)
                    * (alpha.powf(self.exponent) * (self.exponent + 2.0)
                        / (2.0 * std::f32::consts::PI))
            } else {
                Color::zero()
            }
        };
        let diffuse_value = self.diffuse.color(uv) * d_out.z * std::f32::consts::FRAC_1_PI;

        specular_value + diffuse_value
    }

    fn roughness(&self, _uv: &Option<Vector2<f32>>) -> f32 {
        // TODO: Two component material now...
        (2.0 / (2.0 + self.exponent)).sqrt()
    }

    fn is_twosided(&self) -> bool {
        true
    }

    fn bsdf_type(&self) -> BSDFType {
        BSDFType::GLOSSY
    }
    fn bsdf_event(&self) -> BSDFEvent {
        BSDFEvent::REFLECTION
    }
}
