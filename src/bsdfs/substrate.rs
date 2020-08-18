use crate::bsdfs::distribution::*;
use crate::bsdfs::utils::*;
use crate::bsdfs::*;
use crate::math::cosine_sample_hemisphere;
use cgmath::InnerSpace;

// Uses the simpler model (FresnelBlend)
pub struct BSDFSubstrate {
    pub specular: BSDFColor,
    pub diffuse: BSDFColor,
    pub distribution: Option<MicrofacetDistributionBSDF>,
}

impl BSDFSubstrate {
    pub fn schlick_fresnel(&self, uv: &Option<Vector2<f32>>, cos_theta: f32) -> Color {
        let rs = self.specular.color(uv);
        rs + (Color::one() - rs) * (1.0 - cos_theta).powi(5)
    }
}

impl BSDF for BSDFSubstrate {
    fn sample(
        &self,
        uv: &Option<Vector2<f32>>,
        d_in: &Vector3<f32>,
        mut s: Point2<f32>,
        transport: Transport,
    ) -> Option<SampledDirection> {
        if d_in.z <= 0.0 {
            return None;
        };

        // TODO: Importance sampling specular or diffuse part
        let (d_out, domain) = if s.x < 0.5 {
            s.x *= 2.0; // Rescale random number
                        // assert!(s.x >= 0.0 && s.x < 1.0, "Wrong s.x for diffuse {}", s.x);
            let d_out = cosine_sample_hemisphere(s);
            (d_out, Domain::SolidAngle)
        } else {
            s.x = (s.x - 0.5) * 2.0;
            // assert!(s.x >= 0.0 && s.x < 1.0, "Wrong s.x for specular {}", s.x);
            let (m, domain) = match self.distribution {
                None => {
                    // Pure specular object
                    (Vector3::new(0.0, 0.0, 1.0), Domain::Discrete)
                }
                Some(ref d) => {
                    // Microfacet distribution
                    let distr = MicrofacetDistribution {
                        microfacet_type: d.microfacet_type,
                        alpha_u: d.alpha_u,
                        alpha_v: d.alpha_v,
                    };

                    let (m, pdf) = distr.sample(s);
                    if pdf == 0.0 {
                        return None;
                    }

                    (m, Domain::SolidAngle)
                }
            };

            let d_out = reflect_vector(*d_in, m);
            if cos_theta(&d_out) <= 0.0 {
                return None;
            }
            (d_out, domain)
        };

        let pdf = self.pdf(uv, d_in, &d_out, domain, transport);
        if pdf.value() == 0.0 {
            return None;
        }
        let f = self.eval(uv, d_in, &d_out, domain, transport);

        Some(SampledDirection {
            weight: f / pdf.value(),
            d: d_out,
            pdf: pdf,
            eta: 1.0,
        })
    }

    fn pdf(
        &self,
        _uv: &Option<Vector2<f32>>,
        wi: &Vector3<f32>,
        wo: &Vector3<f32>,
        domain: Domain,
        _: Transport,
    ) -> PDF {
        // Check that we got the correct configuration
        if wi.z <= 0.0 || wo.z <= 0.0 {
            return match domain {
                Domain::SolidAngle => PDF::SolidAngle(0.0),
                Domain::Discrete => PDF::Discrete(0.0),
                // _ => panic!("invalid domain"),
            };
        }

        // Compute the half-vector
        let m = wi + wo;
        if m.x == 0.0 && m.y == 0.0 && m.z == 0.0 {
            return match domain {
                Domain::SolidAngle => PDF::SolidAngle(0.0),
                Domain::Discrete => PDF::Discrete(0.0),
                // _ => panic!("invalid domain"),
            };
        }
        let m = m.normalize();

        match domain {
            Domain::Discrete => {
                // Pure specular
                assert!(domain == Domain::Discrete);
                if check_reflection_condition(wi, wo) {
                    PDF::Discrete(0.5)
                } else {
                    // For now, raise an error.
                    unimplemented!();
                }
            }
            Domain::SolidAngle => {
                let pdf_diffuse = wo.z * std::f32::consts::FRAC_1_PI;
                let pdf_specular = match self.distribution {
                    None => 0.0,
                    Some(ref d) => {
                        // Microfacet distribution
                        let distr = MicrofacetDistribution {
                            microfacet_type: d.microfacet_type,
                            alpha_u: d.alpha_u,
                            alpha_v: d.alpha_v,
                        };
                        distr.pdf(&m) / (4.0 * wo.dot(m).abs())
                    }
                };
                PDF::SolidAngle(0.5 * (pdf_diffuse + pdf_specular))
            }
        }
    }

    fn eval(
        &self,
        uv: &Option<Vector2<f32>>,
        d_in: &Vector3<f32>,
        d_out: &Vector3<f32>,
        domain: Domain,
        _: Transport,
    ) -> Color {
        // Check that we got the correct configuration
        if d_in.z <= 0.0 || d_out.z <= 0.0 {
            return Color::zero();
        }

        // Compute the half-vector
        let m = d_in + d_out;
        if m.x == 0.0 && m.y == 0.0 && m.z == 0.0 {
            return Color::zero();
        }
        let m = m.normalize();

        match domain {
            Domain::SolidAngle => {
                let diffuse = self.diffuse.color(uv)
                    * (Color::one() - self.specular.color(uv))
                    * (28.0 / (23.0 * std::f32::consts::PI))
                    * (1.0 - (1.0 - 0.5 * abs_cos_theta(d_in)).powi(5))
                    * (1.0 - (1.0 - 0.5 * abs_cos_theta(&d_out)).powi(5));

                // Compute the specular component if requested
                let specular = match self.distribution {
                    None => Color::zero(),
                    Some(ref d) => {
                        let distr = MicrofacetDistribution {
                            microfacet_type: d.microfacet_type,
                            alpha_u: d.alpha_u,
                            alpha_v: d.alpha_v,
                        };

                        let model = distr.eval(&m)
                            / (4.0
                                * d_in.dot(m).abs()
                                * (cos_theta(d_in).abs().max(cos_theta(&d_out).abs())));
                        let schlick_fresnel = self.schlick_fresnel(uv, d_in.dot(m));
                        model * schlick_fresnel
                    }
                };
                // We return the cosine weighted bsdf
                (diffuse + specular) * d_out.z
            }
            Domain::Discrete => {
                if check_reflection_condition(d_in, d_out) {
                    self.schlick_fresnel(uv, d_in.dot(m))
                } else {
                    // For now, raise an error.
                    unimplemented!();
                }
            } // _ => panic!("Area domain for BSDF")
        }
    }

    fn roughness(&self, _uv: &Option<Vector2<f32>>) -> f32 {
        unimplemented!()
    }

    fn is_smooth(&self) -> bool {
        self.distribution.is_none()
    }
    fn is_twosided(&self) -> bool {
        true
    }
}
