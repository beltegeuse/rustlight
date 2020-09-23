use crate::bsdfs::utils::*;
use crate::bsdfs::*;
use std::collections::HashMap;

// Table from Mitsuba
lazy_static! {
    static ref IOR_DATA: HashMap<&'static str, f32> = {
        let mut m = HashMap::new();
        m.insert("vacuum", 1.0);
        m.insert("helium", 1.000036);
        m.insert("hydrogen", 1.000132);
        m.insert("air", 1.000277);
        m.insert("carbon dioxide", 1.00045);
        m.insert("water", 1.3330);
        m.insert("acetone", 1.36);
        m.insert("ethanol", 1.361);
        m.insert("carbon tetrachloride", 1.461);
        m.insert("glycerol", 1.4729);
        m.insert("benzene", 1.501);
        m.insert("silicone oil", 1.52045);
        m.insert("bromine", 1.661);
        m.insert("water ice", 1.31);
        m.insert("fused quartz", 1.458);
        m.insert("pyrex", 1.470);
        m.insert("acrylic glass", 1.49);
        m.insert("polypropylene", 1.49);
        m.insert("bk7", 1.5046);
        m.insert("sodium chloride", 1.544);
        m.insert("amber", 1.55);
        m.insert("pet", 1.5750);
        m.insert("diamond", 2.419);
        m
    };
}

pub struct BSDFGlass {
    pub specular_transmittance: BSDFColor,
    pub specular_reflectance: BSDFColor,
    pub eta: f32,
    pub inv_eta: f32,
}

impl BSDFGlass {
    pub fn eta(mut self, int_ior: f32, ext_ior: f32) -> Self {
        self.eta = int_ior / ext_ior;
        assert_ne!(self.eta, 0.0);
        self.inv_eta = 1.0 / self.eta;
        self
    }

    fn refract(&self, wi: &Vector3<f32>, cos_theta_t: f32) -> Vector3<f32> {
        let scale = if cos_theta_t < 0.0 {
            -self.inv_eta
        } else {
            -self.eta
        };
        let v = Vector3::new(scale * wi.x, scale * wi.y, cos_theta_t);
        v
    }
}
impl Default for BSDFGlass {
    fn default() -> Self {
        let int_ior = IOR_DATA["bk7"];
        let ext_ior = IOR_DATA["air"];
        Self {
            specular_transmittance: BSDFColor::default(),
            specular_reflectance: BSDFColor::default(),
            eta: 1.0,
            inv_eta: 1.0,
        }
        .eta(int_ior, ext_ior)
    }
}

impl BSDF for BSDFGlass {
    fn sample(
        &self,
        uv: &Option<Vector2<f32>>,
        d_in: &Vector3<f32>,
        s: Point2<f32>,
        transport: Transport,
    ) -> Option<SampledDirection> {
        let (fresnel, cos_theta_trans) = fresnel_dielectric(d_in.z, self.eta);

        // IS the fresnel coefficient
        if s.x <= fresnel {
            // Reflection over the surface
            Some(SampledDirection {
                weight: self.specular_reflectance.color(uv),
                d: reflect(d_in),
                pdf: PDF::Discrete(fresnel),
                eta: 1.0,
                event: BSDFEvent::REFLECTION,
                event_type: BSDFType::DELTA,
            })
        } else {
            // Radiance must be scaled to account for the solid angle compression
            // that occurs when crossing the interface.
            let factor = if transport == Transport::Radiance {
                if cos_theta_trans < 0.0 {
                    self.inv_eta
                } else {
                    self.eta
                }
            } else {
                1.0
            };

            Some(SampledDirection {
                weight: self.specular_transmittance.color(uv) * factor * factor,
                d: self.refract(d_in, cos_theta_trans),
                pdf: PDF::Discrete(fresnel),
                eta: if cos_theta_trans < 0.0 {
                    self.eta
                } else {
                    self.inv_eta
                },
                event: BSDFEvent::TRANSMISSION,
                event_type: BSDFType::DELTA,
            })
        }
    }

    fn pdf(
        &self,
        _uv: &Option<Vector2<f32>>,
        _wi: &Vector3<f32>,
        _wo: &Vector3<f32>,
        domain: Domain,
        _transport: Transport,
    ) -> PDF {
        assert!(domain == Domain::Discrete);
        // Should not be used anyway for now
        todo!()
    }

    fn eval(
        &self,
        uv: &Option<Vector2<f32>>,
        wi: &Vector3<f32>,
        wo: &Vector3<f32>,
        domain: Domain,
        transport: Transport,
    ) -> Color {
        assert!(domain == Domain::Discrete);

        let (fresnel, cos_theta_trans) = fresnel_dielectric(wi.z, self.eta);

        // Depends if we transmit or reflect
        if wi.z * wo.z >= 0.0 {
            // Reflection
            if check_reflection_condition(wi, wo) {
                self.specular_reflectance.color(uv) * fresnel
            } else {
                // For now, raise an error.
                unimplemented!();
            }
        } else {
            let factor = if transport == Transport::Radiance {
                if cos_theta_trans < 0.0 {
                    self.inv_eta
                } else {
                    self.eta
                }
            } else {
                1.0
            };

            if check_direlectric_condition(wi, wo, self.eta, cos_theta_trans) {
                self.specular_transmittance.color(uv) * (1.0 - fresnel) * factor * factor
            } else {
                // For now, raise an error.
                unimplemented!();
            }
        }
    }

    fn roughness(&self, _uv: &Option<Vector2<f32>>) -> f32 {
        0.0
    }

    fn is_twosided(&self) -> bool {
        false
    }

    fn bsdf_type(&self) -> BSDFType {
        BSDFType::DELTA
    }
    fn bsdf_event(&self) -> BSDFEvent {
        BSDFEvent::TRANSMISSION | BSDFEvent::REFLECTION
    }
}
