use cgmath::Point2;
use paths::path::*;
use paths::vertex::*;
use samplers::Sampler;
use scene::Scene;
use std::rc::Rc;
use structure::Color;
use integrators::gradient::explicit::TechniqueGradientPathTracing;

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
    fn base<'a>(
        &mut self,
        technique: &mut TechniqueGradientPathTracing,
        pos: Point2<u32>,
        scene: &'a Scene,
        sampler: &mut Sampler,
    ) -> (Color, Rc<VertexPtr<'a>>);
    fn shift<'a>(
        &mut self,
        technique: &mut TechniqueGradientPathTracing,
        pos: Point2<u32>,
        scene: &Scene,
        sampler: &mut Sampler,
        base: &Rc<VertexPtr<'a>>,
    ) -> ShiftValue;
    fn clear(&mut self);
}

pub mod diffuse;
pub mod random_replay;
