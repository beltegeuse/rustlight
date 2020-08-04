use crate::bsdfs::*;

pub struct BSDFBlend {
    pub bsdf1: Box<dyn BSDF + Sync + Send>,
    pub bsdf2: Box<dyn BSDF + Sync + Send>,
    pub weight: f32,
}

impl BSDF for BSDFBlend {
    fn sample(
        &self,
        uv: &Option<Vector2<f32>>,
        d_in: &Vector3<f32>,
        sample: Point2<f32>,
        transport: Transport,
    ) -> Option<SampledDirection> {
        assert!(!self.bsdf1.is_smooth() && !self.bsdf2.is_smooth());

        // Select the BSDF proportional to their respective weights
        let sampled_dir = if sample.x < self.weight {
            let scaled_sample = Point2::new(sample.x * (1.0 / self.weight), sample.y);
            self.bsdf1.sample(uv, d_in, scaled_sample, transport)
        } else {
            let scaled_sample = Point2::new(
                (sample.x - self.weight) * (1.0 / (1.0 - self.weight)),
                sample.y,
            );
            self.bsdf2.sample(uv, d_in, scaled_sample, transport)
        };

        if let Some(mut sampled_dir) = sampled_dir {
            sampled_dir.pdf = self.pdf(uv, d_in, &sampled_dir.d, Domain::SolidAngle, transport);
            if sampled_dir.pdf.value() == 0.0 {
                None
            } else {
                sampled_dir.weight =
                    self.eval(uv, d_in, &sampled_dir.d, Domain::SolidAngle, transport)
                        / sampled_dir.pdf.value();
                Some(sampled_dir)
            }
        } else {
            None
        }
    }

    fn pdf(
        &self,
        uv: &Option<Vector2<f32>>,
        d_in: &Vector3<f32>,
        d_out: &Vector3<f32>,
        domain: Domain,
        transport: Transport,
    ) -> PDF {
        let pdf_1 = self.bsdf1.pdf(uv, d_in, d_out, domain, transport) * self.weight;
        let pdf_2 = self.bsdf2.pdf(uv, d_in, d_out, domain, transport) * (1.0 - self.weight);
        if let (PDF::SolidAngle(pdf_1), PDF::SolidAngle(pdf_2)) = (pdf_1, pdf_2) {
            PDF::SolidAngle(pdf_1 + pdf_2)
        } else {
            panic!("get wrong type of BSDF");
        }
    }

    fn eval(
        &self,
        uv: &Option<Vector2<f32>>,
        d_in: &Vector3<f32>,
        d_out: &Vector3<f32>,
        domain: Domain,
        transport: Transport,
    ) -> Color {
        debug_assert!(self.weight >= 0.0 && self.weight <= 1.0);
        self.weight * self.bsdf1.eval(uv, d_in, d_out, domain, transport)
            + (1.0 - self.weight) * self.bsdf2.eval(uv, d_in, d_out, domain, transport)
    }

    fn roughness(&self, uv: &Option<Vector2<f32>>) -> f32 {
        // TODO: Use a more finer scheme when multiple component
        // BSDF will be implemented
        self.bsdf1.roughness(uv).min(self.bsdf2.roughness(uv))
    }

    fn is_smooth(&self) -> bool {
        if self.bsdf1.is_smooth() || self.bsdf2.is_smooth() {
            panic!("is smooth on blend material");
        }
        false
    }

    fn is_twosided(&self) -> bool {
        if !self.bsdf1.is_twosided() || !self.bsdf2.is_twosided() {
            panic!("is twosided on blend material");
        }
        true
    }
}
