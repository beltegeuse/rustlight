use crate::emitter::Emitter;
use crate::scene::*;
use crate::structure::*;
use cgmath::*;
use std::cell::RefCell;
use std::rc::{Rc, Weak};

#[derive(Clone)]
pub struct Edge {
    /// Geometric informations
    pub dist: Option<f32>,
    pub d: Vector3<f32>,
    /// Connecting two vertices
    pub vertices: (VertexID, Option<VertexID>),
    /// Sampling information
    pub pdf_distance: f32,
    pub pdf_direction: PDF,
    pub weight: Color,
    pub rr_weight: f32,
    pub id_sampling: usize,
}

impl Edge {
    pub fn from_vertex(
        path: &mut Path,
        org_vertex_id: VertexID,
        pdf_direction: PDF,
        weight: Color,
        rr_weight: f32,
        next_vertex_id: VertexID,
        id_sampling: usize,
    ) -> EdgeID {
        let mut d = path.vertex(next_vertex_id).position() - path.vertex(org_vertex_id).position();
        let dist = d.magnitude();
        d /= dist;

        let edge = Edge {
            dist: Some(dist),
            d,
            vertices: (org_vertex_id, Some(next_vertex_id)),
            pdf_distance: 1.0,
            pdf_direction,
            weight,
            rr_weight,
            id_sampling,
        };
        let edge = path.register_edge(edge);

        match path.vertex_mut(next_vertex_id) {
            Vertex::Light(ref mut v) => v.edge_in = Some(edge),
            _ => unimplemented!(),
        };
        edge
    }

    pub fn from_ray<'scene>(
        path: &mut Path<'scene, '_>,
        ray: &Ray,
        org_vertex_id: VertexID,
        pdf_direction: PDF,
        weight: Color,
        rr_weight: f32,
        scene: &'scene Scene,
        id_sampling: usize,
    ) -> (EdgeID, Option<VertexID>) {
        // TODO: When there will be volume, we need to sample a distance inside the volume
        let edge = Edge {
            dist: None,
            d: ray.d,
            vertices: (org_vertex_id, None),
            pdf_distance: 1.0,
            pdf_direction,
            weight,
            rr_weight,
            id_sampling,
        };
        let edge = path.register_edge(edge);
        let its = match scene.trace(&ray) {
            Some(its) => its,
            None => {
                // Create an edge without distance
                return (edge, None);
            }
        };

        // Create the new vertex
        let intersection_distance = its.dist;
        let new_vertex = Vertex::Surface(SurfaceVertex {
            its,
            rr_weight: 1.0,
            edge_in: edge,
            edge_out: vec![],
        });
        let new_vertex = path.register_vertex(new_vertex);

        // Update the edge information
        {
            let edge = path.edge_mut(edge);
            edge.dist = Some(intersection_distance);
            edge.vertices.1 = Some(new_vertex);
        }
        (edge, Some(new_vertex))
    }

    pub fn next_on_light_source(&self, path: &Path) -> bool {
        if let Some(v) = &self.vertices.1 {
            return path.vertex(*v).on_light_source();
        } else {
            return false; //TODO: No env map
        }
    }

    /// Get the contribution along this edge (toward the light direction)
    /// @deprecated: This might be not optimal as it is not a recursive call.
    pub fn contribution(&self, path: &Path) -> Color {
        if let Some(v) = &self.vertices.1 {
            return self.weight * self.rr_weight * path.vertex(*v).contribution(self);
        } else {
            return Color::zero(); //TODO: No env map
        }
    }
}

#[derive(Clone)]
pub struct SensorVertex {
    pub uv: Point2<f32>,
    pub pos: Point3<f32>,
    pub edge_in: Option<EdgeID>,
    pub edge_out: Option<EdgeID>,
}

#[derive(Clone)]
pub struct SurfaceVertex<'scene> {
    pub its: Intersection<'scene>,
    pub rr_weight: f32,
    pub edge_in: EdgeID,
    pub edge_out: Vec<EdgeID>,
}

#[derive(Clone)]
pub struct EmitterVertex<'emitter> {
    pub pos: Point3<f32>,
    pub n: Vector3<f32>,
    pub emitter: &'emitter dyn Emitter,
    pub edge_in: Option<EdgeID>,
    pub edge_out: Option<EdgeID>,
}

#[derive(Clone)]
pub enum Vertex<'scene, 'emitter> {
    Sensor(SensorVertex),
    Surface(SurfaceVertex<'scene>),
    Light(EmitterVertex<'emitter>),
}
impl<'scene, 'emitter> Vertex<'scene, 'emitter> {
    pub fn pixel_pos(&self) -> Point2<f32> {
        match *self {
            Vertex::Sensor(ref v) => v.uv,
            _ => unreachable!(),
        }
    }
    pub fn position(&self) -> Point3<f32> {
        match *self {
            Vertex::Surface(ref v) => v.its.p,
            Vertex::Sensor(ref v) => v.pos,
            Vertex::Light(ref v) => v.pos,
        }
    }

    pub fn on_light_source(&self) -> bool {
        match *self {
            Vertex::Surface(ref v) => !v.its.mesh.emission.is_zero(),
            Vertex::Sensor(ref _v) => false,
            Vertex::Light(ref _v) => true,
        }
    }

    pub fn contribution(&self, edge: &Edge) -> Color {
        match *self {
            Vertex::Surface(ref v) => {
                if v.its.n_s.dot(-edge.d) >= 0.0 {
                    v.its.mesh.emission
                } else {
                    Color::zero()
                }
            }
            Vertex::Sensor(ref _v) => Color::zero(),
            Vertex::Light(ref v) => v.emitter.emitted_luminance(-edge.d), // FIXME: Check the normal orientation
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VertexID(usize);
#[derive(Clone, Copy, Debug)]
pub struct EdgeID(usize);
pub struct Path<'scene, 'emitter> {
    vertices: Vec<Vertex<'scene, 'emitter>>,
    edges: Vec<Edge>,
}
impl<'scene, 'emitter> Default for Path<'scene, 'emitter> {
    fn default() -> Self {
        Path {
            vertices: vec![],
            edges: vec![],
        }
    }
}
impl<'scene, 'emitter> Path<'scene, 'emitter> {
    pub fn register_edge(&mut self, e: Edge) -> EdgeID {
        let id = self.edges.len();
        self.edges.push(e);
        EdgeID(id)
    }
    pub fn register_vertex(&mut self, v: Vertex<'scene, 'emitter>) -> VertexID {
        let id = self.vertices.len();
        self.vertices.push(v);
        VertexID(id)
    }
    pub fn vertex(&self, id: VertexID) -> &Vertex<'scene, 'emitter> {
        &self.vertices[id.0]
    }
    pub fn edge(&self, id: EdgeID) -> &Edge {
        &self.edges[id.0]
    }
    pub fn vertex_mut(&mut self, id: VertexID) -> &mut Vertex<'scene, 'emitter> {
        &mut self.vertices[id.0]
    }
    pub fn edge_mut(&mut self, id: EdgeID) -> &mut Edge {
        &mut self.edges[id.0]
    }

    // pub fn next_vertex(&self) -> Vec<Rc<VertexPtr<'a>>> {
    //     match *self {
    //         Vertex::Sensor(ref v) => match v.edge_out.as_ref() {
    //             None => vec![],
    //             Some(ref e) => e
    //                 .borrow()
    //                 .vertices
    //                 .1
    //                 .as_ref()
    //                 .map_or(vec![], |v| vec![v.clone()]),
    //         },
    //         _ => unimplemented!(),
    //     }
    // }
}
