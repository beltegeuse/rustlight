use image::*;
use serde::{Deserialize, Deserializer};
use serde_json;
use structure::*;

use cgmath::{Point2, Vector2, Vector3};

use pbrt_rs;
use std;
use tools::*;

// Texture or uniform color buffers
#[derive(Deserialize)]
pub struct Texture {
    #[serde(deserialize_with = "deserialize_from_str")]
    pub img: DynamicImage,
}

impl Texture {
    pub fn pixel(&self, mut uv: Vector2<f32>) -> Color {
        uv.x = uv.x.modulo(1.0);
        uv.y = uv.y.modulo(1.0);

        let dim = self.img.dimensions();
        let (x, y) = (uv.x * dim.0 as f32, uv.y * dim.1 as f32);
        let pix = self.img.get_pixel(x as u32, y as u32);
        assert!(pix.data[3] == 255); // Just check that there is no alpha
        Color::new(
            f32::from(pix.data[0]) / 255.0,
            f32::from(pix.data[1]) / 255.0,
            f32::from(pix.data[2]) / 255.0,
        )
    }
}

fn deserialize_from_str<'de, D>(deserializer: D) -> Result<DynamicImage, D::Error>
where
    D: Deserializer<'de>,
{
    let _s: String = Deserialize::deserialize(deserializer)?;
    let _img = DynamicImage::new_rgb8(1, 1);
    unimplemented!();
    // Ok(_img)
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
    fn pdf(&self, uv: &Option<Vector2<f32>>, d_in: &Vector3<f32>, d_out: &Vector3<f32>) -> PDF;
    /// eval the bsdf value : $fs(...)$
    fn eval(&self, uv: &Option<Vector2<f32>>, d_in: &Vector3<f32>, d_out: &Vector3<f32>) -> Color;
    /// check if it is smooth
    //TODO: Replace this using flags
    fn is_smooth(&self) -> bool;
    fn is_twosided(&self) -> bool;
}

pub mod diffuse;
pub mod distribution;
pub mod metal;
pub mod phong;
pub mod specular;

use bsdfs::diffuse::BSDFDiffuse;
use bsdfs::distribution::TrowbridgeReitzDistribution;
use bsdfs::metal::BSDFMetal;
use bsdfs::phong::BSDFPhong;
use bsdfs::specular::BSDFSpecular;

/// Dispatch coded BSDF
pub fn parse_bsdf(
    b: &serde_json::Value,
) -> Result<Box<BSDF + Send + Sync>, Box<std::error::Error>> {
    let new_bsdf_type: String = serde_json::from_value(b["type"].clone())?;
    let new_bsdf: Box<BSDF + Send + Sync> = match new_bsdf_type.as_ref() {
        "phong" => Box::<BSDFPhong>::new(serde_json::from_value(b["data"].clone())?),
        "diffuse" => Box::<BSDFDiffuse>::new(serde_json::from_value(b["data"].clone())?),
        "specular" => Box::<BSDFSpecular>::new(serde_json::from_value(b["data"].clone())?),
        _ => panic!("Unknown BSDF type {}", new_bsdf_type),
    };
    Ok(new_bsdf)
}

pub fn bsdf_texture_match(v: &pbrt_rs::Param) -> Option<BSDFColor> {
    match v {
        pbrt_rs::Param::RGB(r, g, b) => Some(BSDFColor::UniformColor(Color::new(*r, *g, *b))),
        _ => None,
    }
}

pub fn bsdf_pbrt(bsdf: &pbrt_rs::BSDF) -> Box<BSDF + Sync + Send> {
    let bsdf: Option<Box<BSDF + Sync + Send>> = match bsdf {
        pbrt_rs::BSDF::Matte(ref v) => {
            if let Some(diffuse) = bsdf_texture_match(&v.kd) {
                Some(Box::new(BSDFDiffuse { diffuse }))
            } else {
                None
            }
        }
        pbrt_rs::BSDF::Metal(ref v) => {
            let eta = bsdf_texture_match(&v.eta).unwrap();
            let k = bsdf_texture_match(&v.k).unwrap();
            let u_roughness = 0.1;//bsdf_texture_match(&v.u_roughness.as_ref().unwrap()).unwrap();
            let v_roughness = 0.1;//bsdf_texture_match(&v.v_roughness.as_ref().unwrap()).unwrap();
            // TODO: roughness is ignored
            // FIXME: remap
            Some(Box::new(BSDFMetal {
                r: BSDFColor::UniformColor(Color::value(1.0)),
                distribution: TrowbridgeReitzDistribution::new(
                    u_roughness, //u_roughness.color(&None).r,
                    v_roughness, //v_roughness.color(&None).r,
                    true,
                ),
                k,
                eta_i: BSDFColor::UniformColor(Color::one()),
                eta_t: eta,
            }))
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
