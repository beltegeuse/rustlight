use bsdfs::*;

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

    fn pdf(&self, _uv: &Option<Vector2<f32>>, _: &Vector3<f32>, _: &Vector3<f32>) -> PDF {
        PDF::Discrete(1.0)
    }

    fn eval(&self, _uv: &Option<Vector2<f32>>, _: &Vector3<f32>, _: &Vector3<f32>) -> Color {
        // For now, we do not implement this function
        // as we want to avoid to call this function
        // and does not handle correctly the evaluation
        unimplemented!()
    }

    fn is_smooth(&self) -> bool {
        false
    }
}
