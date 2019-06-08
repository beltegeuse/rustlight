use crate::integrators::gradient::explicit::TechniqueGradientPathTracing;
use crate::paths::path::*;
use crate::paths::vertex::*;
use crate::samplers::Sampler;
use crate::scene::*;
use crate::structure::Color;
use cgmath::Point2;

/// Shift mapping definition
pub struct ShiftValue {
    pub base: Color,
    pub offset: Color,
    pub gradient: Color,
}
impl Default for ShiftValue {
    fn default() -> Self {
        ShiftValue {
            base: Color::zero(),
            offset: Color::zero(),
            gradient: Color::zero(),
        }
    }
}
impl ShiftValue {
    pub fn base(mut self, base: Color) -> Self {
        self.base = base;
        self
    }
}
pub trait ShiftMapping {
    fn base<'scene, 'emitter>(
        &mut self,
        path: &mut Path<'scene, 'emitter>,
        technique: &mut TechniqueGradientPathTracing,
        pos: Point2<u32>,
        scene: &'scene Scene,
        emitters: &'emitter EmitterSampler,
        sampler: &mut Sampler,
    ) -> (Color, VertexID);
    fn shift<'scene, 'emitter>(
        &mut self,
        path: &mut Path<'scene, 'emitter>,
        technique: &mut TechniqueGradientPathTracing,
        pos: Point2<u32>,
        scene: &'scene Scene,
        emitters: &'emitter EmitterSampler,
        sampler: &mut Sampler,
        base: VertexID,
    ) -> ShiftValue;
    fn clear(&mut self);
}

pub mod diffuse;
pub mod random_replay;
