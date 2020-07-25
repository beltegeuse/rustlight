use crate::emitter::Emitter;
use crate::paths::edge::*;
use crate::paths::path::*;
use crate::structure::*;
use crate::volume::*;
use cgmath::*;

#[derive(Clone)]
pub enum Vertex<'scene, 'emitter> {
    Sensor {
        uv: Point2<f32>,
        pos: Point3<f32>,
        edge_in: Option<EdgeID>,
        edge_out: Option<EdgeID>,
    },
    Surface {
        its: Intersection<'scene>,
        rr_weight: f32,
        edge_in: EdgeID,
        edge_out: Vec<EdgeID>,
    },
    Light {
        pos: Point3<f32>,
        n: Vector3<f32>,
        emitter: &'emitter dyn Emitter,
        edge_in: Option<EdgeID>,
        edge_out: Option<EdgeID>,
    },
    Volume {
        phase_function: PhaseFunction,
        pos: Point3<f32>,
        d_in: Vector3<f32>,
        rr_weight: f32,
        edge_in: EdgeID,
        edge_out: Vec<EdgeID>,
    },
}
impl<'scene, 'emitter> Vertex<'scene, 'emitter> {
    pub fn pixel_pos(&self) -> Point2<f32> {
        match *self {
            Vertex::Sensor { uv, .. } => uv,
            _ => unreachable!(),
        }
    }
    pub fn position(&self) -> Point3<f32> {
        match self {
            Vertex::Surface { its, .. } => its.p,
            Vertex::Sensor { pos, .. } | Vertex::Light { pos, .. } | Vertex::Volume { pos, .. } => {
                *pos
            }
        }
    }
    pub fn on_surface(&self) -> bool {
        match *self {
            Vertex::Surface { .. } | Vertex::Light { .. } => true,
            Vertex::Sensor { .. } | Vertex::Volume { .. } => false,
        }
    }
    pub fn on_light_source(&self) -> bool {
        match self {
            Vertex::Surface { its, .. } => !its.mesh.emission.is_zero(),
            Vertex::Sensor { .. } | Vertex::Volume { .. } => false,
            Vertex::Light { .. } => true,
        }
    }

    pub fn contribution(&self, edge: &Edge) -> Color {
        match self {
            Vertex::Surface { its, .. } => {
                if its.n_s.dot(-edge.d) >= 0.0 {
                    its.mesh.emission
                } else {
                    Color::zero()
                }
            }
            Vertex::Sensor { .. } | Vertex::Volume { .. } => Color::zero(),
            Vertex::Light { emitter, .. } => emitter.emitted_luminance(-edge.d), // FIXME: Check the normal orientation
        }
    }
}
