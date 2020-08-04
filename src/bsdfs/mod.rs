use crate::structure::*;
use serde::{Deserialize, Deserializer};
use serde_json;

use cgmath::{Point2, Vector2, Vector3};
#[cfg(feature = "pbrt")]
use pbrt_rs;
use std;

// Texture or uniform color buffers
#[derive(Deserialize)]
pub struct Texture {
    #[cfg(feature = "image")]
    #[serde(deserialize_with = "deserialize_from_str")]
    pub img: Bitmap,
}

impl Texture {
    // With features
    #[cfg(feature = "image")]
    pub fn load(path: &str) -> Texture {
        Texture {
            img: Bitmap::read(path),
        }
    }
    #[cfg(feature = "image")]
    pub fn pixel(&self, uv: Vector2<f32>) -> Color {
        self.img.pixel_uv(uv)
    }

    // Without
    #[cfg(not(feature = "image"))]
    pub fn load(_path: &str) -> Texture {
        unimplemented!("No support of textures");
    }
    #[cfg(not(feature = "image"))]
    pub fn pixel(&self, _uv: Vector2<f32>) -> Color {
        unimplemented!("No support of images");
    }
}

#[cfg(feature = "image")]
fn deserialize_from_str<'de, D>(deserializer: D) -> Result<Bitmap, D::Error>
where
    D: Deserializer<'de>,
{
    let _s: String = Deserialize::deserialize(deserializer)?;
    unimplemented!();
}

#[derive(Deserialize)]
pub enum BSDFColor {
    UniformColor(Color),
    TextureColor(Texture), // FIXME
}

impl BSDFColor {
    pub fn color(&self, uv: &Option<Vector2<f32>>) -> Color {
        match self {
            BSDFColor::UniformColor(ref c) => *c,
            BSDFColor::TextureColor(ref t) => {
                if let Some(uv_coords) = uv {
                    t.pixel(*uv_coords)
                } else {
                    warn!("Found a texture but no uv coordinate given");
                    Color::zero()
                }
            }
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

/// Dispatch coded BSDF
pub fn parse_bsdf(
    b: &serde_json::Value,
) -> Result<Box<dyn BSDF + Send + Sync>, Box<dyn std::error::Error>> {
    let new_bsdf_type: String = serde_json::from_value(b["type"].clone())?;
    let new_bsdf: Box<dyn BSDF + Send + Sync> = match new_bsdf_type.as_ref() {
        "phong" => Box::<BSDFPhong>::new(serde_json::from_value(b["data"].clone())?),
        "diffuse" => Box::<BSDFDiffuse>::new(serde_json::from_value(b["data"].clone())?),
        "metal" => Box::<BSDFMetal>::new(serde_json::from_value(b["data"].clone())?),
        _ => panic!("Unknown BSDF type {}", new_bsdf_type),
    };
    Ok(new_bsdf)
}

#[cfg(feature = "pbrt")]
fn bsdf_texture_match(v: &pbrt_rs::Param, scene_info: &pbrt_rs::Scene) -> Option<BSDFColor> {
    match v {
        pbrt_rs::Param::Float(ref v) => {
            if v.len() != 1 {
                panic!("Impossible to build textureColor with: {:?}", v);
            }
            let v = v[0];
            Some(BSDFColor::UniformColor(Color::new(v, v, v)))
        }
        pbrt_rs::Param::RGB(ref rgb) => {
            Some(BSDFColor::UniformColor(Color::new(rgb.r, rgb.g, rgb.b)))
        }
        pbrt_rs::Param::Name(ref name) => {
            if let Some(texture) = scene_info.textures.get(name) {
                Some(BSDFColor::TextureColor(Texture::load(&texture.filename)))
            } else {
                warn!("Impossible to found an texture with name: {}", name);
                None
            }
        }
        _ => None,
    }
}

// Debug macro for color
// macro_rules! default_color {
//     ($texture: expr, $default:expr) => {{
//         if let Some(v) = $texture {
//             v
//         } else {
//             BSDFColor::UniformColor($default)
//         }
//     }};
// }

#[cfg(feature = "pbrt")]
pub fn bsdf_pbrt(bsdf: &pbrt_rs::BSDF, scene_info: &pbrt_rs::Scene) -> Box<dyn BSDF + Sync + Send> {
    let bsdf: Option<Box<dyn BSDF + Sync + Send>> = match bsdf {
        pbrt_rs::BSDF::Matte(ref v) => {
            if let Some(diffuse) = bsdf_texture_match(&v.kd, scene_info) {
                Some(Box::new(BSDFDiffuse { diffuse }))
            } else {
                None
            }
        }
        pbrt_rs::BSDF::Glass(ref v) => {
            // Get BSDF colors
            let specular_reflectance = bsdf_texture_match(&v.kr, scene_info).unwrap();
            let specular_transmittance = bsdf_texture_match(&v.kt, scene_info).unwrap();

            // Roughness
            let u_roughness = bsdf_texture_match(&v.u_roughness, scene_info).unwrap();
            let v_roughness = bsdf_texture_match(&v.v_roughness, scene_info).unwrap();

            // FIXME: be able to load float textures?
            let (u_roughness, v_roughness) =
                (u_roughness.color(&None).r, v_roughness.color(&None).r);
            if u_roughness != 0.0 || v_roughness != 0.0 {
                warn!(
                    "roughness found for glass material ({}, {})",
                    u_roughness, v_roughness
                );
            }

            let index = bsdf_texture_match(&v.index, scene_info).unwrap();
            let eta = match index {
                BSDFColor::UniformColor(v) => v.r,
                _ => unimplemented!("Texture ETA is not supported"),
            };

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
        pbrt_rs::BSDF::Metal(ref v) => {
            let eta = bsdf_texture_match(&v.eta, scene_info).unwrap();
            let k = bsdf_texture_match(&v.k, scene_info).unwrap();
            let (u_roughness, v_roughness) = if let (Some(ref u_rough), Some(ref v_rough)) =
                (v.u_roughness.as_ref(), v.v_roughness.as_ref())
            {
                (
                    bsdf_texture_match(u_rough, scene_info).unwrap(),
                    bsdf_texture_match(v_rough, scene_info).unwrap(),
                )
            } else {
                (
                    bsdf_texture_match(&v.roughness, scene_info).unwrap(),
                    bsdf_texture_match(&v.roughness, scene_info).unwrap(),
                )
            };
            // FIXME: be able to load float textures?
            let (u_roughness, v_roughness) =
                (u_roughness.color(&None).r, v_roughness.color(&None).r);

            // FIXME: Do we need to remap???
            assert_eq!(u_roughness, v_roughness);

            Some(Box::new(BSDFMetal {
                specular: BSDFColor::UniformColor(Color::one()),
                eta,
                k,
                distribution: Some(MicrofacetDistributionBSDF {
                    microfacet_type: MicrofacetType::GGX,
                    alpha_u: u_roughness,
                    alpha_v: v_roughness,
                }),
            }))
        }
        pbrt_rs::BSDF::Mirror(ref v) => {
            let specular = bsdf_texture_match(&v.kr, scene_info).unwrap();
            Some(Box::new(BSDFMetal {
                specular,
                eta: BSDFColor::UniformColor(Color::one()),
                k: BSDFColor::UniformColor(Color::zero()),
                distribution: None,
            }))
        }
        pbrt_rs::BSDF::Substrate(ref v) => {
            let kd = bsdf_texture_match(&v.kd, scene_info).unwrap();
            let ks = bsdf_texture_match(&v.ks, scene_info).unwrap();
            let u_roughness = bsdf_texture_match(&v.u_roughness, scene_info).unwrap();
            let v_roughness = bsdf_texture_match(&v.v_roughness, scene_info).unwrap();

            // FIXME: be able to load float textures?
            let (u_roughness, v_roughness) =
                (u_roughness.color(&None).r, v_roughness.color(&None).r);
            assert_eq!(u_roughness, v_roughness);
            let distribution = if u_roughness != 0.0 {
                Some(MicrofacetDistributionBSDF {
                    microfacet_type: MicrofacetType::GGX,
                    alpha_u: u_roughness,
                    alpha_v: v_roughness,
                })
            } else {
                None
            };

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
