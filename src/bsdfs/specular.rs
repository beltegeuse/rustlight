use crate::bsdfs::*;

#[derive(Deserialize)]
pub struct BSDFSpecular {
    pub specular: BSDFColor,
}

impl BSDF for BSDFSpecular {
    fn sample(
        &self,
        uv: &Option<Vector2<f32>>,
        d_in: &Vector3<f32>,
        _: Point2<f32>,
    ) -> Option<SampledDirection> {
        if d_in.z <= 0.0 {
            None
        } else {
            Some(SampledDirection {
                weight: self.specular.color(uv),
                d: reflect(d_in),
                pdf: PDF::Discrete(1.0),
            })
        }
    }

    fn pdf(
        &self,
        _uv: &Option<Vector2<f32>>,
        wi: &Vector3<f32>,
        wo: &Vector3<f32>,
        domain: Domain,
    ) -> PDF {
        assert!(domain == Domain::Discrete);
        if check_reflection_condition(wi, wo) {
            PDF::Discrete(1.0)
        } else {
            // For now, raise an error.
            unimplemented!();
        }
    }

    fn eval(
        &self,
        uv: &Option<Vector2<f32>>,
        wi: &Vector3<f32>,
        wo: &Vector3<f32>,
        domain: Domain,
    ) -> Color {
        assert!(domain == Domain::Discrete);
        if check_reflection_condition(wi, wo) {
            self.specular.color(uv)
        } else {
            // For now, raise an error.
            unimplemented!();
        }
    }

    fn roughness(&self, _uv: &Option<Vector2<f32>>) -> f32 {
        0.0
    }

    fn is_smooth(&self) -> bool {
        true
    }
    fn is_twosided(&self) -> bool {
        true
    }
}
