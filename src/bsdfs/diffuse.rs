use crate::bsdfs::*;
use crate::math::cosine_sample_hemisphere;
use std;

#[derive(Deserialize)]
pub struct BSDFDiffuse {
    pub diffuse: BSDFColor,
}

impl BSDF for BSDFDiffuse {
    fn sample(
        &self,
        uv: &Option<Vector2<f32>>,
        d_in: &Vector3<f32>,
        sample: Point2<f32>,
    ) -> Option<SampledDirection> {
        if d_in.z <= 0.0 {
            None
        } else {
            let d_out = cosine_sample_hemisphere(sample);
            Some(SampledDirection {
                weight: self.diffuse.color(uv),
                d: d_out,
                pdf: PDF::SolidAngle(d_out.z * std::f32::consts::FRAC_1_PI),
            })
        }
    }

    fn pdf(
        &self,
        _uv: &Option<Vector2<f32>>,
        d_in: &Vector3<f32>,
        d_out: &Vector3<f32>,
        domain: Domain,
    ) -> PDF {
        assert!(domain == Domain::SolidAngle);

        if d_in.z <= 0.0 {
            return PDF::SolidAngle(0.0);
        }
        if d_out.z <= 0.0 {
            PDF::SolidAngle(0.0)
        } else {
            PDF::SolidAngle(d_out.z * std::f32::consts::FRAC_1_PI)
        }
    }

    fn eval(
        &self,
        uv: &Option<Vector2<f32>>,
        d_in: &Vector3<f32>,
        d_out: &Vector3<f32>,
        domain: Domain,
    ) -> Color {
        assert!(domain == Domain::SolidAngle);

        if d_in.z <= 0.0 {
            return Color::zero();
        }
        if d_out.z > 0.0 {
            self.diffuse.color(uv) * d_out.z * std::f32::consts::FRAC_1_PI
        } else {
            Color::zero()
        }
    }

    fn roughness(&self, _uv: &Option<Vector2<f32>>) -> f32 {
        std::f32::INFINITY
    }

    fn is_smooth(&self) -> bool {
        false
    }
    fn is_twosided(&self) -> bool {
        true
    }
}
