use cgmath::*;
use paths::path::*;
use paths::vertex::*;
use samplers::*;
use scene::*;
use structure::*;

#[derive(Clone)]
pub struct SurfaceVertexShift<'a> {
    pub its: Intersection<'a>,
    pub pdf_bsdf: Option<PDF>,
    /// Containts the throughput times the jacobian
    pub throughput: Color,
    /// Contains the ratio of PDF (with the Jacobian embeeded)
    pub pdf_ratio: f32,
}

pub enum ShiftVertex<'a> {
    Sensor(SensorVertex),
    Surface(SurfaceVertexShift<'a>),
}

pub struct ShiftPath<'a> {
    pub vertices: Vec<ShiftVertex<'a>>,
    pub edges: Vec<Edge>,
}

pub trait ShiftOp<'a> {
    fn generate_base<S: Sampler>(
        &mut self,
        pix: (u32, u32),
        scene: &'a Scene,
        sampler: &mut S,
        max_depth: Option<u32>,
    ) -> Option<Path<'a>>;
    fn shift<S: Sampler>(
        &mut self,
        base_path: &Path<'a>,
        shift_pix: Point2<f32>,
        scene: &'a Scene,
        sampler: &mut S,
        max_depth: Option<u32>,
    ) -> Option<ShiftPath<'a>>;
}
