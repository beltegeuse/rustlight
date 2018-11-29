use bsdfs::distribution::*;
use bsdfs::*;
use cgmath::InnerSpace;

pub struct BSDFMetal {
    pub r: BSDFColor,
    pub distribution: TrowbridgeReitzDistribution,
    // Fresnel
    pub k: BSDFColor,
    pub eta_i: BSDFColor,
    pub eta_t: BSDFColor,
}

pub fn fr_conductor(cos_theta_i: f32, eta_i: Color, eta_t: Color, k: Color) -> Color {
    let not_clamped: f32 = cos_theta_i;
    let cos_theta_i: f32 = clamp_t(not_clamped, -1.0, 1.0);
    let eta: Color = eta_t / eta_i;
    let eta_k: Color = k / eta_i;
    let cos_theta_i2: f32 = cos_theta_i * cos_theta_i;
    let sin_theta_i2: f32 = 1.0 as f32 - cos_theta_i2;
    let eta_2: Color = eta * eta;
    let eta_k2: Color = eta_k * eta_k;
    let t0: Color = eta_2 - eta_k2 - Color::value(sin_theta_i2);
    let a2_plus_b2: Color = (t0 * t0 + eta_2 * eta_k2 * Color::value(4 as f32)).sqrt();
    let t1: Color = a2_plus_b2 + Color::value(cos_theta_i2);
    let a: Color = ((a2_plus_b2 + t0) * 0.5 as f32).sqrt();
    let t2: Color = a * 2.0 as f32 * cos_theta_i;
    let rs: Color = (t1 - t2) / (t1 + t2);
    let t3: Color = a2_plus_b2 * cos_theta_i2 + Color::value(sin_theta_i2 * sin_theta_i2);
    let t4: Color = t2 * sin_theta_i2;
    let rp: Color = rs * (t3 - t4) / (t3 + t4);
    (rp + rs) * Color::value(0.5)
}

impl BSDF for BSDFMetal {
    fn sample(
        &self,
        uv: &Option<Vector2<f32>>,
        d_in: &Vector3<f32>,
        sample: Point2<f32>,
    ) -> Option<SampledDirection> {
        if d_in.z == 0.0 {
            return None;
        }
        let wh = self.distribution.sample_wh(d_in, &sample);
        let d = reflect_vector(*d_in, wh);
        if !vec3_same_hemisphere_vec3(d_in, &d) {
            return None;
        }
        // compute PDF of _wi_ for microfacet reflection
        let pdf = self.distribution.pdf(d_in, &wh) / (4.0 * d_in.dot(wh));
        let weight = self.eval(uv, d_in, &d, Domain::SolidAngle);
        Some(SampledDirection {
            weight: weight / pdf,
            d,
            pdf: PDF::SolidAngle(pdf),
        })
    }

    fn pdf(&self, _uv: &Option<Vector2<f32>>, d_in: &Vector3<f32>, d_out: &Vector3<f32>, domain: Domain) -> PDF {
        assert!(domain == Domain::SolidAngle);

        if !vec3_same_hemisphere_vec3(d_out, d_in) {
            return PDF::SolidAngle(0.0);
        }
        let wh = (d_out + d_in).normalize();
        PDF::SolidAngle(self.distribution.pdf(d_in, &wh) / (4.0 * d_in.dot(wh)))
    }

    fn eval(&self, uv: &Option<Vector2<f32>>, d_in: &Vector3<f32>, d_out: &Vector3<f32>, domain: Domain) -> Color {
        assert!(domain == Domain::SolidAngle);
        
        let cos_theta_o = d_out.z;
        let cos_theta_i = d_in.z;
        if cos_theta_o <= 0.0 {
            return Color::zero();
        }
        if cos_theta_i <= 0.0 {
            return Color::zero();
        }
        let wh = *d_out + *d_in;
        // handle degenerate cases for microfacet reflection
        if wh.x == 0.0 && wh.y == 0.0 && wh.z == 0.0 {
            return Color::zero();
        }
        let wh = wh.normalize();
        let f: Color = fr_conductor(
            d_out.dot(wh),
            self.eta_i.color(uv),
            self.eta_t.color(uv),
            self.k.color(uv),
        );
        self.r.color(uv) * self.distribution.d(&wh) * self.distribution.g(d_in, d_out) * f
            / (4.0 * cos_theta_i) // * cos_theta_o
    }

    fn is_smooth(&self) -> bool {
        false
    }
    fn is_twosided(&self) -> bool {
        true
    }
}
