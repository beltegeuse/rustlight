use crate::paths::edge::*;
use crate::paths::vertex::*;
use crate::samplers::Sampler;
use crate::scene::Scene;
use crate::structure::Color;
use cgmath::Point2;

#[derive(Clone, Copy, Debug)]
pub struct VertexID(usize);
#[derive(Clone, Copy, Debug)]
pub struct EdgeID(usize);

/// Path structure
/// This looks like a pool
pub struct Path<'scene> {
    vertices: Vec<Vertex<'scene>>,
    edges: Vec<Edge>,
}
impl<'scene, 'emitter> Default for Path<'scene> {
    fn default() -> Self {
        Path {
            vertices: vec![],
            edges: vec![],
        }
    }
}
impl<'scene, 'emitter> Path<'scene> {
    /// Clear path (to be able to reuse it)
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.edges.clear();
    }

    /// Generate a root path from a light
    pub fn from_light(
        &mut self,
        scene: &'scene Scene,
        sampler: &mut dyn Sampler,
    ) -> (VertexID, Color) {
        let (emitter, sampled_point, flux) = scene.emitters().random_sample_emitter_position(
            sampler.next(),
            sampler.next(),
            sampler.next2d(),
        );
        let emitter_vertex = Vertex::Light {
            pos: sampled_point.p,
            n: sampled_point.n,
            uv: sampled_point.uv,
            primitive_id: sampled_point.primitive_id,
            emitter,
            edge_in: None,
            edge_out: None,
        };
        (self.register_vertex(emitter_vertex), flux)
    }
    pub fn from_sensor(
        &mut self,
        img_pos: Point2<u32>,
        scene: &Scene,
        sampler: &mut dyn Sampler,
    ) -> (VertexID, Color) {
        // Only generate a path from the sensor
        let root = Vertex::Sensor {
            uv: Point2::new(
                img_pos.x as f32 + sampler.next(),
                img_pos.y as f32 + sampler.next(),
            ),
            pos: scene.camera.position(),
            edge_in: None,
            edge_out: None,
        };
        return (self.register_vertex(root), Color::one());
    }

    pub fn register_edge(&mut self, e: Edge) -> EdgeID {
        let id = self.edges.len();
        self.edges.push(e);
        EdgeID(id)
    }
    pub fn register_vertex(&mut self, v: Vertex<'scene>) -> VertexID {
        let id = self.vertices.len();
        self.vertices.push(v);
        VertexID(id)
    }
    pub fn vertex(&self, id: VertexID) -> &Vertex<'scene> {
        &self.vertices[id.0]
    }
    pub fn edge(&self, id: EdgeID) -> &Edge {
        &self.edges[id.0]
    }
    pub fn vertex_mut(&mut self, id: VertexID) -> &mut Vertex<'scene> {
        &mut self.vertices[id.0]
    }
    pub fn edge_mut(&mut self, id: EdgeID) -> &mut Edge {
        &mut self.edges[id.0]
    }
    pub fn have_next_vertices(&self, vertex_id: VertexID) -> bool {
        !self.next_vertices(vertex_id).is_empty()
    }
    pub fn next_vertices(&self, vertex_id: VertexID) -> Vec<(EdgeID, VertexID)> {
        let mut next_vertices = vec![];
        match self.vertex(vertex_id) {
            Vertex::Surface { edge_out, .. } | Vertex::Volume { edge_out, .. } => {
                for edge_id in edge_out {
                    let edge = self.edge(*edge_id);
                    if let Some(vertex_next_id) = edge.vertices.1 {
                        next_vertices.push((*edge_id, vertex_next_id));
                    }
                }
            }
            Vertex::Sensor { edge_out, .. } | Vertex::Light { edge_out, .. } => {
                if let Some(edge_id) = edge_out {
                    let edge = self.edge(*edge_id);
                    if let Some(vertex_next_id) = edge.vertices.1 {
                        next_vertices.push((*edge_id, vertex_next_id));
                    }
                }
            }
        }
        next_vertices
    }
}
