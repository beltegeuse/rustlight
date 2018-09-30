use bsdfs::distribution::*;
use bsdfs::*;
use cgmath::InnerSpace;

pub struct BSDFGlass {
    pub kr: BSDFColor,
    pub ks: BSDFColor,
    pub distribution: TrowbridgeReitzDistribution,
    pub index: f32,
}

