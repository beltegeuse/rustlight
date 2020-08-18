use crate::structure::*;

use cgmath::{Point2, Vector2, Vector3};
#[cfg(feature = "mitsuba")]
use mitsuba_rs;
#[cfg(feature = "pbrt")]
use pbrt_rs;
use std;
use std::collections::HashMap;

// Texture or uniform color buffers
pub struct Texture {
    pub img: Bitmap,
}

impl Texture {
    // Load from a file
    pub fn load(path: &str) -> Texture {
        Texture {
            img: Bitmap::read(path),
        }
    }
}

pub enum BSDFColor {
    UniformColor(Color),
    TextureColor(Texture),
}

impl BSDFColor {
    pub fn color(&self, uv: &Option<Vector2<f32>>) -> Color {
        match self {
            BSDFColor::UniformColor(ref c) => *c,
            BSDFColor::TextureColor(ref t) => {
                if let Some(uv_coords) = uv {
                    t.img.pixel_uv(*uv_coords)
                } else {
                    warn!("Found a texture but no uv coordinate given");
                    Color::zero()
                }
            }
        }
    }

    pub fn avg(&self) -> Color {
        match self {
            BSDFColor::UniformColor(c) => *c,
            BSDFColor::TextureColor(c) => c.img.average(),
        }
    }
}
impl Default for BSDFColor {
    fn default() -> Self {
        BSDFColor::UniformColor(Color::one())
    }
}

// Helpers
fn reflect(d: &Vector3<f32>) -> Vector3<f32> {
    Vector3::new(-d.x, -d.y, d.z)
}

/// Struct that represent a sampled direction
#[derive(Clone)]
pub struct SampledDirection {
    pub weight: Color,
    pub d: Vector3<f32>,
    pub pdf: PDF,
    pub eta: f32,
}

pub trait BSDF: Send + Sync {
    /// sample an random direction based on the BSDF value
    /// @d_in: the incomming direction in the local space
    /// @sample: random number 2D
    /// @return: the outgoing direction, the pdf and the bsdf value $fs(...) * | n . d_out |$
    fn sample(
        &self,
        uv: &Option<Vector2<f32>>,
        d_in: &Vector3<f32>,
        sample: Point2<f32>,
        transport: Transport,
    ) -> Option<SampledDirection>;
    /// eval the bsdf pdf value in solid angle
    fn pdf(
        &self,
        uv: &Option<Vector2<f32>>,
        d_in: &Vector3<f32>,
        d_out: &Vector3<f32>,
        domain: Domain,
        transport: Transport,
    ) -> PDF;
    /// eval the bsdf value : $fs(...)$
    fn eval(
        &self,
        uv: &Option<Vector2<f32>>,
        d_in: &Vector3<f32>,
        d_out: &Vector3<f32>,
        domain: Domain,
        transport: Transport,
    ) -> Color;
    /// return the roughness of the material
    fn roughness(&self, uv: &Option<Vector2<f32>>) -> f32;
    /// check if it is smooth
    //TODO: Replace this using flags
    fn is_smooth(&self) -> bool;
    /// Used to automatically flip the normal vector
    fn is_twosided(&self) -> bool;
}

pub mod blend;
pub mod diffuse;
pub mod distribution;
pub mod glass;
pub mod metal;
pub mod phong;
pub mod substrate;
pub mod utils;

use crate::bsdfs::diffuse::BSDFDiffuse;
use crate::bsdfs::distribution::*;
use crate::bsdfs::glass::BSDFGlass;
use crate::bsdfs::metal::BSDFMetal;
use crate::bsdfs::phong::BSDFPhong;
use crate::bsdfs::substrate::BSDFSubstrate;

#[cfg(feature = "pbrt")]
fn bsdf_texture_match_pbrt(
    v: &pbrt_rs::parser::Spectrum,
    textures: &HashMap<String, pbrt_rs::Texture>,
) -> Option<BSDFColor> {
    match v {
        pbrt_rs::parser::Spectrum::RGB(rgb) => {
            Some(BSDFColor::UniformColor(Color::new(rgb.r, rgb.g, rgb.b)))
        }
        pbrt_rs::parser::Spectrum::Texture(name) => {
            if let Some(texture) = textures.get(name) {
                Some(BSDFColor::TextureColor(Texture::load(&texture.filename)))
            } else {
                warn!("Impossible to found an texture with name: {}", name);
                None
            }
        }
        _ => None,
    }
}

#[cfg(feature = "pbrt")]
fn bsdf_texture_f32_match_pbrt(
    v: &pbrt_rs::parser::BSDFFloat,
    _textures: &HashMap<String, pbrt_rs::Texture>,
) -> f32 {
    match v {
        pbrt_rs::parser::BSDFFloat::Float(v) => *v,
        pbrt_rs::parser::BSDFFloat::Texture(_name) => panic!("Float texture are not suppoerted"),
    }
}

#[cfg(feature = "pbrt")]
fn distribution_pbrt(
    d: &pbrt_rs::Distribution,
    textures: &HashMap<String, pbrt_rs::Texture>,
) -> MicrofacetDistributionBSDF {
    warn!("Remap roughness is ignored!");
    match &d.roughness {
        pbrt_rs::Roughness::Isotropic(v) => {
            let alpha = bsdf_texture_f32_match_pbrt(v, textures);
            MicrofacetDistributionBSDF {
                microfacet_type: MicrofacetType::GGX,
                alpha_u: alpha,
                alpha_v: alpha,
            }
        }
        pbrt_rs::Roughness::Anisotropic { u, v } => {
            let alpha_u = bsdf_texture_f32_match_pbrt(u, textures);
            let alpha_v = bsdf_texture_f32_match_pbrt(v, textures);
            MicrofacetDistributionBSDF {
                microfacet_type: MicrofacetType::GGX,
                alpha_u,
                alpha_v,
            }
        }
    }
}

#[cfg(feature = "pbrt")]
pub fn bsdf_pbrt(
    bsdf: &pbrt_rs::BSDF,
    textures: &HashMap<String, pbrt_rs::Texture>,
) -> Box<dyn BSDF + Sync + Send> {
    let bsdf: Option<Box<dyn BSDF + Sync + Send>> = match bsdf {
        pbrt_rs::BSDF::Matte { kd, .. } => {
            if let Some(diffuse) = bsdf_texture_match_pbrt(kd, textures) {
                Some(Box::new(BSDFDiffuse { diffuse }))
            } else {
                None
            }
        }
        pbrt_rs::BSDF::Glass {
            kr,
            kt,
            distribution,
            eta,
            ..
        } => {
            // Get BSDF colors
            let specular_reflectance = bsdf_texture_match_pbrt(kr, textures).unwrap();
            let specular_transmittance = bsdf_texture_match_pbrt(kt, textures).unwrap();

            // TODO
            if distribution.is_some() {
                warn!("Glass distribution is ignored. Pure glass instead");
            }

            let eta = bsdf_texture_f32_match_pbrt(eta, textures);

            // FIXME: Bumpmapping is not supported!
            Some(Box::new(
                BSDFGlass {
                    specular_transmittance,
                    specular_reflectance,
                    eta: 1.0,
                    inv_eta: 1.0,
                }
                .eta(eta, 1.0),
            ))
        }
        pbrt_rs::BSDF::Metal {
            eta,
            k,
            distribution,
            ..
        } => {
            let eta = bsdf_texture_match_pbrt(eta, textures).unwrap();
            let k = bsdf_texture_match_pbrt(k, textures).unwrap();
            let distribution = Some(distribution_pbrt(distribution, textures));
            Some(Box::new(BSDFMetal {
                specular: BSDFColor::UniformColor(Color::one()),
                eta,
                k,
                distribution,
            }))
        }
        pbrt_rs::BSDF::Mirror { kr, .. } => {
            let specular = bsdf_texture_match_pbrt(kr, textures).unwrap();
            Some(Box::new(BSDFMetal {
                specular,
                eta: BSDFColor::UniformColor(Color::one()),
                k: BSDFColor::UniformColor(Color::zero()),
                distribution: None,
            }))
        }
        pbrt_rs::BSDF::Substrate {
            kd,
            ks,
            distribution,
            ..
        } => {
            let kd = bsdf_texture_match_pbrt(kd, textures).unwrap();
            let ks = bsdf_texture_match_pbrt(ks, textures).unwrap();
            let distribution = Some(distribution_pbrt(distribution, textures));
            Some(Box::new(BSDFSubstrate {
                specular: ks,
                diffuse: kd,
                distribution,
            }))
        } // _ => None,
    };

    if let Some(bsdf) = bsdf {
        bsdf
    } else {
        Box::new(BSDFDiffuse {
            diffuse: BSDFColor::UniformColor(Color::value(0.8)),
        })
    }
}

#[cfg(feature = "mitsuba")]
fn bsdf_texture_match_mts(v: &mitsuba_rs::BSDFColorSpectrum, wk: &std::path::Path) -> BSDFColor {
    match v {
        mitsuba_rs::BSDFColorSpectrum::Constant(v) => {
            let v = v.clone().as_rgb();
            let v = Color {
                r: v.r,
                g: v.g,
                b: v.b,
            };
            BSDFColor::UniformColor(v)
        }
        mitsuba_rs::BSDFColorSpectrum::Texture(v) => {
            let mut texture = Texture::load(wk.join(v.filename.clone()).to_str().unwrap());
            if v.gamma != 1.0 {
                texture.img.gamma(1.0 / v.gamma);
            }
            BSDFColor::TextureColor(texture)
        }
    }
}

#[cfg(feature = "mitsuba")]
fn bsdf_texture_f32_mts(v: &mitsuba_rs::BSDFColorFloat, _wk: &std::path::Path) -> f32 {
    match v {
        mitsuba_rs::BSDFColorFloat::Constant(v) => *v,
        _ => panic!("Float texture are not supported yet!"),
    }
}

#[cfg(feature = "mitsuba")]
fn distribution_mts(
    d: &Option<mitsuba_rs::Distribution>,
    wk: &std::path::Path,
) -> Option<MicrofacetDistributionBSDF> {
    match d {
        None => None,
        Some(d) => {
            let (alpha_u, alpha_v) = match &d.alpha {
                mitsuba_rs::Alpha::Isotropic(alpha) => {
                    let alpha = bsdf_texture_f32_mts(&alpha, wk);
                    (alpha, alpha)
                }
                mitsuba_rs::Alpha::Anisotropic { u, v } => {
                    let alpha_u = bsdf_texture_f32_mts(&u, wk);
                    let alpha_v = bsdf_texture_f32_mts(&v, wk);
                    assert_eq!(alpha_u, alpha_v); // No anisotropic material for now
                    (alpha_u, alpha_v)
                }
            };

            let microfacet_type = match &d.distribution[..] {
                "beckmann" => MicrofacetType::Beckmann,
                "ggx" => MicrofacetType::GGX,
                _ => panic!("Unsupported microfacet type {}", d.distribution),
            };

            Some(MicrofacetDistributionBSDF {
                microfacet_type,
                alpha_u,
                alpha_v,
            })
        }
    }
}

#[cfg(feature = "mitsuba")]
pub fn bsdf_mts(bsdf: &mitsuba_rs::BSDF, wk: &std::path::Path) -> Box<dyn BSDF + Sync + Send> {
    let bsdf: Option<Box<dyn BSDF + Sync + Send>> = match bsdf {
        mitsuba_rs::BSDF::TwoSided { bsdf } => {
            // Rustlight automatically apply twosided
            Some(bsdf_mts(&bsdf, wk))
        }
        mitsuba_rs::BSDF::Diffuse { reflectance } => {
            let diffuse = bsdf_texture_match_mts(reflectance, wk);
            Some(Box::new(BSDFDiffuse { diffuse }))
        }
        mitsuba_rs::BSDF::Phong {
            exponent,
            specular_reflectance,
            diffuse_reflectance
        } => {
            let specular = bsdf_texture_match_mts(specular_reflectance, wk);
            let diffuse = bsdf_texture_match_mts(diffuse_reflectance, wk);
            let exponent = bsdf_texture_f32_mts(exponent, &wk);

            let weight_specular = {
                let d_avg = diffuse.avg().luminance();
                let s_avg = specular.avg().luminance();
                assert!(d_avg + s_avg != 0.0);
                s_avg / (d_avg + s_avg)
            };

            Some(Box::new(BSDFPhong {
                diffuse,
                specular,
                exponent,
                weight_specular
            }))
        }
        // Thin material are ignored
        // Impossible to do rough glass
        mitsuba_rs::BSDF::Dielectric {
            int_ior,
            ext_ior,
            specular_reflectance,
            specular_transmittance,
            ..
        } => {
            // TODO: Implement distribution
            let specular_reflectance = bsdf_texture_match_mts(specular_reflectance, wk);
            let specular_transmittance = bsdf_texture_match_mts(specular_transmittance, wk);
            Some(Box::new(
                BSDFGlass {
                    specular_transmittance,
                    specular_reflectance,
                    eta: 1.0,
                    inv_eta: 1.0,
                }
                .eta(*int_ior, *ext_ior),
            ))
        }
        // TODO: Might be a mismatch between different BSDF
        mitsuba_rs::BSDF::Plastic {
            distribution,
            specular_reflectance,
            diffuse_reflectance,
            .. // Ignoring nonlinear, int_ior and ext_ior
        } => {
            let specular = bsdf_texture_match_mts(specular_reflectance, wk);
            let diffuse = bsdf_texture_match_mts(diffuse_reflectance, wk);

            Some(Box::new(
                BSDFSubstrate {
                    specular,
                    diffuse,
                    distribution: distribution_mts(distribution, &wk)
                }
            ))
        }
        mitsuba_rs::BSDF::Conductor {
            distribution, eta, k, ext_eta, specular_reflectance
        } => {
            let specular = bsdf_texture_match_mts(specular_reflectance, wk);
            let eta = {
                let eta = eta.clone().as_rgb();
                BSDFColor::UniformColor( Color {
                    r: eta.r / ext_eta,
                    g: eta.g / ext_eta,
                    b: eta.b / ext_eta,
                })
            };
            let k = {
                let k = k.clone().as_rgb();
                BSDFColor::UniformColor( Color {
                    r: k.r / ext_eta,
                    g: k.g / ext_eta,
                    b: k.b / ext_eta,
                })
            };

            Some(Box::new(
                BSDFMetal {
                    specular,
                    eta,
                    k,
                    distribution: distribution_mts(distribution, wk)
                }
            ))
        }
        _ => None,
    };

    if let Some(bsdf) = bsdf {
        bsdf
    } else {
        Box::new(BSDFDiffuse {
            diffuse: BSDFColor::UniformColor(Color::value(0.8)),
        })
    }
}
