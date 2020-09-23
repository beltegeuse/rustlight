use crate::bsdfs::distribution::*;
use crate::bsdfs::utils::*;
use crate::bsdfs::*;
use cgmath::InnerSpace;

pub struct BSDFMetal {
    pub specular: BSDFColor,
    /// Real and imaginary material component
    pub eta: BSDFColor,
    pub k: BSDFColor,
    pub distribution: Option<MicrofacetDistributionBSDF>,
}

impl BSDF for BSDFMetal {
    fn sample(
        &self,
        uv: &Option<Vector2<f32>>,
        d_in: &Vector3<f32>,
        s: Point2<f32>,
        _: Transport,
    ) -> Option<SampledDirection> {
        if d_in.z <= 0.0 {
            None
        } else {
            match self.distribution {
                None => {
                    // Pure specular object
                    Some(SampledDirection {
                        weight: self.specular.color(uv)
                            * fresnel_conductor(d_in.z, self.eta.color(uv), self.k.color(uv)),
                        d: reflect(d_in),
                        pdf: PDF::Discrete(1.0),
                        eta: 1.0,
                        event: BSDFEvent::REFLECTION,
                        event_type: BSDFType::DELTA,
                    })
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

                    let wo = reflect_vector(*d_in, m);
                    if cos_theta(&wo) <= 0.0 {
                        return None;
                    }

                    let f = fresnel_conductor(d_in.dot(m), self.eta.color(uv), self.k.color(uv))
                        * self.specular.color(uv);

                    // TODO: Need to be different if we sample the weights
                    let w = distr.eval(&m) * distr.g(d_in, &wo, &m) * d_in.dot(m)
                        / (pdf * cos_theta(d_in));

                    Some(SampledDirection {
                        weight: w * f,
                        d: wo,
                        pdf: PDF::SolidAngle(pdf),
                        eta: 1.0,
                        event: BSDFEvent::REFLECTION,
                        event_type: BSDFType::GLOSSY,
                    })
                }
            }
        }
    }

    fn pdf(
        &self,
        _uv: &Option<Vector2<f32>>,
        wi: &Vector3<f32>,
        wo: &Vector3<f32>,
        domain: Domain,
        _: Transport,
    ) -> PDF {
        match self.distribution {
            None => {
                // Pure specular
                assert!(domain == Domain::Discrete);
                if check_reflection_condition(wi, wo) {
                    PDF::Discrete(1.0)
                } else {
                    // For now, raise an error.
                    unimplemented!();
                }
            }
            Some(ref d) => {
                /* Calculate the reflection half-vector */
                let h = (wi + wo).normalize();

                // Microfacet distribution
                let distr = MicrofacetDistribution {
                    microfacet_type: d.microfacet_type,
                    alpha_u: d.alpha_u,
                    alpha_v: d.alpha_v,
                };

                // FIXME: Visibility
                PDF::SolidAngle(distr.pdf(&h) / (4.0 * wo.dot(h).abs()))
            }
        }
    }

    fn eval(
        &self,
        uv: &Option<Vector2<f32>>,
        wi: &Vector3<f32>,
        wo: &Vector3<f32>,
        domain: Domain,
        _: Transport,
    ) -> Color {
        match self.distribution {
            None => {
                assert!(domain == Domain::Discrete);
                if check_reflection_condition(wi, wo) {
                    self.specular.color(uv)
                        * fresnel_conductor(wi.z.abs(), self.eta.color(uv), self.k.color(uv))
                } else {
                    // For now, raise an error.
                    unimplemented!();
                }
            }
            Some(ref d) => {
                /* Calculate the reflection half-vector */
                let h = (wi + wo).normalize();

                // Microfacet distribution
                let distr = MicrofacetDistribution {
                    microfacet_type: d.microfacet_type,
                    alpha_u: d.alpha_u,
                    alpha_v: d.alpha_v,
                };

                let d = distr.eval(&h);
                if d == 0.0 {
                    return Color::zero();
                }

                /* Fresnel factor */
                let f = self.specular.color(uv)
                    * fresnel_conductor(wi.dot(h), self.eta.color(uv), self.k.color(uv));
                /* Smith's shadow-masking function */
                let g = distr.g(wi, wo, &h);
                /* Calculate the total amount of reflection */
                let model = d * g / (4.0 * cos_theta(wi));

                f * model
            }
        }
    }

    fn roughness(&self, _uv: &Option<Vector2<f32>>) -> f32 {
        unimplemented!()
    }

    fn is_twosided(&self) -> bool {
        true
    }

    fn bsdf_type(&self) -> BSDFType {
        match self.distribution {
            Some(_) => BSDFType::GLOSSY,
            None => BSDFType::DELTA,
        }
    }
    fn bsdf_event(&self) -> BSDFEvent {
        BSDFEvent::REFLECTION
    }
}
