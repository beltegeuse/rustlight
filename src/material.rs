use cgmath::{Point2, Vector3};
use cgmath::InnerSpace;
use math::{cosine_sample_hemisphere, Frame};
use serde_json;
use std;
use structure::*;

// Helpers
fn reflect(d: &Vector3<f32>) -> Vector3<f32> {
    Vector3::new(-d.x, -d.y, d.z)
}

/// Dispatch coded BSDF
pub fn parse_bsdf(b: &serde_json::Value) -> Result<Box<BSDF + Send + Sync>, Box<std::error::Error>> {
    let new_bsdf_type: String = serde_json::from_value(b["type"].clone())?;
    let new_bsdf: Box<BSDF + Send + Sync> = match new_bsdf_type.as_ref() {
        "phong" => Box::<BSDFPhong>::new(serde_json::from_value(b["data"].clone())?),
        "diffuse" => Box::<BSDFDiffuse>::new(serde_json::from_value(b["data"].clone())?),
        "specular" => Box::<BSDFSpecular>::new(serde_json::from_value(b["data"].clone())?),
        _ => panic!("Unknown BSDF type {}", new_bsdf_type),
    };
    Ok(new_bsdf)
}

/// Struct that represent a sampled direction
#[derive(Clone)]
pub struct SampledDirection {
    pub weight: Color,
    pub d: Vector3<f32>,
    pub pdf: PDF,
}

pub trait BSDF {
    /// sample an random direction based on the BSDF value
    /// @d_in: the incomming direction in the local space
    /// @sample: random number 2D
    /// @return: the outgoing direction, the pdf and the bsdf value $fs(...) * | n . d_out |$
    fn sample(&self, d_in: &Vector3<f32>, sample: Point2<f32>) -> Option<SampledDirection>;
    /// eval the bsdf pdf value in solid angle
    fn pdf(&self, d_in: &Vector3<f32>, d_out: &Vector3<f32>) -> PDF;
    /// eval the bsdf value : $fs(...)$
    fn eval(&self, d_in: &Vector3<f32>, d_out: &Vector3<f32>) -> Color;
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BSDFDiffuse {
    pub diffuse: Color,
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
                pdf: PDF::SolidAngle(d_out.z * std::f32::consts::FRAC_1_PI),
            })
        }
    }

    fn pdf(&self, d_in: &Vector3<f32>, d_out: &Vector3<f32>) -> PDF {
        if d_in.z <= 0.0 { return PDF::SolidAngle(0.0); }
        if d_out.z <= 0.0 { PDF::SolidAngle(0.0) } else { PDF::SolidAngle(d_out.z * std::f32::consts::FRAC_1_PI) }
    }

    fn eval(&self, d_in: &Vector3<f32>, d_out: &Vector3<f32>) -> Color {
        if d_in.z <= 0.0 { return Color::zero(); }
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
        let sin_alpha = (1.0 - sample.y.powf(2.0 / (self.exponent + 1.0))).sqrt();
        let cos_alpha = sample.y.powf(1.0 / (self.exponent + 1.0));
        let phi = 2.0 * std::f32::consts::PI * sample.x;
        let local_dir = Vector3::new(sin_alpha * phi.cos(), sin_alpha * phi.sin(), cos_alpha);

        let frame = Frame::new(reflect(d_in));
        let d_out = frame.to_world(local_dir);
        if d_out.z <= 0.0 {
            None
        } else {
            let pdf = self.pdf(d_in, &d_out);
            if pdf.is_zero() {
                None
            } else {
                Some(SampledDirection {
                    weight: self.eval(d_in, &d_out) / pdf.value(),
                    d: d_out,
                    pdf,
                })
            }
        }
    }

    fn pdf(&self, d_in: &Vector3<f32>, d_out: &Vector3<f32>) -> PDF {
        if d_in.z <= 0.0 { return PDF::SolidAngle(0.0); }
        if d_out.z <= 0.0 {
            PDF::SolidAngle(0.0)
        } else {
            let alpha = reflect(d_in).dot(*d_out);
            if alpha > 0.0 {
                PDF::SolidAngle(alpha.powf(self.exponent) * (self.exponent + 1.0) / (2.0 * std::f32::consts::PI))
            } else {
                PDF::SolidAngle(0.0)
            }
        }
    }

    fn eval(&self, d_in: &Vector3<f32>, d_out: &Vector3<f32>) -> Color {
        if d_in.z <= 0.0 { return Color::zero(); }
        if d_out.z <= 0.0 {
            Color::zero()
        } else {
            let alpha = reflect(d_in).dot(*d_out);
            if alpha > 0.0 {
                self.specular *
                    (alpha.powf(self.exponent) * (self.exponent + 2.0) / (2.0 * std::f32::consts::PI))
            } else {
                Color::zero()
            }
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BSDFSpecular {
    pub specular: Color,
}

impl BSDF for BSDFSpecular {
    fn sample(&self, d_in: &Vector3<f32>, sample: Point2<f32>) -> Option<SampledDirection> {
        if d_in.z <= 0.0 { None } else {
            Some(SampledDirection {
                weight: self.specular,
                d: reflect(d_in),
                pdf: PDF::Discrete(1.0),
            })
        }
    }

    fn pdf(&self, d_in: &Vector3<f32>, d_out: &Vector3<f32>) -> PDF {
        PDF::Discrete(1.0)
    }

    fn eval(&self, d_in: &Vector3<f32>, d_out: &Vector3<f32>) -> Color {
        // For now, we do not implement this function
        // as we want to avoid to call this function
        // and does not handle correctly the evaluation
        unimplemented!()
    }
}
