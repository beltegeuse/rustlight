use crate::paths::edge::*;
use crate::paths::vertex::*;

#[derive(Clone, Copy, Debug)]
pub struct VertexID(usize);
#[derive(Clone, Copy, Debug)]
pub struct EdgeID(usize);

/// Path structure
/// This looks like a pool
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
