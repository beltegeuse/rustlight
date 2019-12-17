use crate::structure::*;
use serde::{Deserialize, Deserializer};
use serde_json;

use cgmath::{InnerSpace, Point2, Vector2, Vector3};
#[cfg(feature = "pbrt")]
use pbrt_rs;
use std;

pub fn reflect_vector(wo: Vector3<f32>, n: Vector3<f32>) -> Vector3<f32> {
    -(wo) + n * 2.0 * wo.dot(n)
}
pub fn check_reflection_condition(wi: &Vector3<f32>, wo: &Vector3<f32>) -> bool {
    return (wi.z * wo.z - wi.x * wo.x - wi.y * wo.y - 1.0).abs() < 0.0001;
}
pub fn check_direlectric_condition(
    wi: &Vector3<f32>,
    wo: &Vector3<f32>,
    eta: f32,
    cos_theta: f32,
) -> bool {
    let dot_p= -wi.x * wo.x * eta - wi.y * wo.y * eta - cos_theta.copysign(wi.z) * wo.z;
    return (dot_p - 1.0).abs() < 0.0001;
}
// Texture or uniform color buffers
#[derive(Deserialize)]
pub struct Texture {
    #[serde(deserialize_with = "deserialize_from_str")]
    pub img: Bitmap,
}

impl Texture {
    pub fn load(path: &str) -> Texture {
        Texture {
            img: Bitmap::read(path),
        }
    }
    // Access to the texture
    pub fn pixel(&self, uv: Vector2<f32>) -> Color {
        self.img.pixel_uv(uv)
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
    ) -> Option<SampledDirection>;
    /// eval the bsdf pdf value in solid angle
    fn pdf(
        &self,
        uv: &Option<Vector2<f32>>,
        d_in: &Vector3<f32>,
        d_out: &Vector3<f32>,
        domain: Domain,
    ) -> PDF;
    /// eval the bsdf value : $fs(...)$
    fn eval(
        &self,
        uv: &Option<Vector2<f32>>,
        d_in: &Vector3<f32>,
        d_out: &Vector3<f32>,
        domain: Domain,
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
pub mod phong;
pub mod specular;

use crate::bsdfs::diffuse::BSDFDiffuse;
use crate::bsdfs::phong::BSDFPhong;
use crate::bsdfs::specular::BSDFSpecular;

/// Dispatch coded BSDF
pub fn parse_bsdf(
    b: &serde_json::Value,
) -> Result<Box<dyn BSDF + Send + Sync>, Box<dyn std::error::Error>> {
    let new_bsdf_type: String = serde_json::from_value(b["type"].clone())?;
    let new_bsdf: Box<dyn BSDF + Send + Sync> = match new_bsdf_type.as_ref() {
        "phong" => Box::<BSDFPhong>::new(serde_json::from_value(b["data"].clone())?),
        "diffuse" => Box::<BSDFDiffuse>::new(serde_json::from_value(b["data"].clone())?),
        "specular" => Box::<BSDFSpecular>::new(serde_json::from_value(b["data"].clone())?),
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
        pbrt_rs::BSDF::Metal(ref v) => {
            let _eta = bsdf_texture_match(&v.eta, scene_info).unwrap();
            let _k = bsdf_texture_match(&v.k, scene_info).unwrap();
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
            assert!(u_roughness != 0.0);
            assert!(v_roughness != 0.0);
            // FIXME: remap
            // Some(Box::new(BSDFMetal {
            //     r: BSDFColor::UniformColor(Color::value(1.0)),
            //     distribution: TrowbridgeReitzDistribution::new(u_roughness, v_roughness, true),
            //     k,
            //     eta_i: BSDFColor::UniformColor(Color::one()),
            //     eta_t: eta,
            // }))
            unimplemented!();
        }
        pbrt_rs::BSDF::Mirror(ref v) => {
            let specular = bsdf_texture_match(&v.kr, scene_info).unwrap();
            Some(Box::new(BSDFSpecular { specular }))
        }
        pbrt_rs::BSDF::Substrate(ref v) => {
            let _kd = bsdf_texture_match(&v.kd, scene_info).unwrap();
            let _ks = bsdf_texture_match(&v.ks, scene_info).unwrap();
            let u_roughness = bsdf_texture_match(&v.u_roughness, scene_info).unwrap();
            let v_roughness = bsdf_texture_match(&v.v_roughness, scene_info).unwrap();
            // FIXME: be able to load float textures?
            let (u_roughness, v_roughness) =
                (u_roughness.color(&None).r, v_roughness.color(&None).r);
            assert!(u_roughness != 0.0);
            assert!(v_roughness != 0.0);

            // let metal = Box::new(BSDFMetal {
            //     r: BSDFColor::UniformColor(Color::value(1.0)),
            //     distribution: TrowbridgeReitzDistribution::new(u_roughness, v_roughness, true),
            //     k: ks,
            //     eta_i: BSDFColor::UniformColor(Color::one()),
            //     eta_t: BSDFColor::UniformColor(Color::one()),
            // });
            // let diffuse = Box::new(BSDFDiffuse { diffuse: kd });
            // Some(Box::new(BSDFBlend {
            //     bsdf1: metal,
            //     bsdf2: diffuse,
            // }))
            unimplemented!();
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
