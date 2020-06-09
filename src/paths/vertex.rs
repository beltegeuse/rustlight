use crate::emitter::Emitter;
use crate::samplers::*;
use crate::accel::*;
use crate::structure::*;
use crate::volume::*;
use cgmath::*;

#[derive(Clone)]
pub struct Edge {
    /// Geometric informations
    pub dist: Option<f32>, // distance between points
    pub d: Vector3<f32>, // edge direction
    /// Connecting two vertices
    pub vertices: (VertexID, Option<VertexID>),
    /// Sampling information (from the BSDF or Phase function)
    pub sampled_distance: Option<SampledDistance>,
    pub pdf_direction: PDF,
    pub weight: Color, // BSDF * Transmittance
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

        // Sampled distance is None here
        // as the distance inside the participating media
        // have not been sampled
        let edge = Edge {
            dist: Some(dist),
            d,
            vertices: (org_vertex_id, Some(next_vertex_id)),
            sampled_distance: None,
            pdf_direction,
            weight,
            rr_weight,
            id_sampling,
        };
        let edge = path.register_edge(edge);

        // This constructor have been only design for light vertex creation
        match path.vertex_mut(next_vertex_id) {
            Vertex::Light { edge_in, .. } => (*edge_in) = Some(edge),
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
        sampler: &mut dyn Sampler,
        accel: &'scene dyn Acceleration,
        medium: Option<&HomogenousVolume>,
        id_sampling: usize,
    ) -> (EdgeID, Option<VertexID>) {
        let edge = Edge {
            dist: None,
            d: ray.d,
            vertices: (org_vertex_id, None),
            sampled_distance: None,
            pdf_direction,
            weight,
            rr_weight,
            id_sampling,
        };
        let edge = path.register_edge(edge);
        let its = match accel.trace(&ray) {
            Some(its) => its,
            None => {
                if let Some(ref m) = medium {
                    // Sample the participating media
                    let mrec = m.sample(ray, sampler.next2d());
                    let pos = Point3::from_vec(ray.o.to_vec() + ray.d * mrec.t);
                    // We are sure to suceed as the distance is infine...
                    // TODO: Note that this design decision makes the env map incompatible with participating media presence
                    assert_eq!(mrec.exited, false);
                    let new_vertex = Vertex::Volume {
                        phase_function: PhaseFunction::Isotropic(),
                        pos,
                        d_in: -ray.d,
                        rr_weight: 1.0,
                        edge_in: edge,
                        edge_out: vec![],
                    };
                    let new_vertex = path.register_vertex(new_vertex);

                    // Update the edge
                    {
                        let edge = path.edge_mut(edge);
                        edge.dist = Some(mrec.t);
                        edge.vertices.1 = Some(new_vertex);
                        edge.weight *= mrec.w;
                        edge.sampled_distance = Some(mrec);
                    }

                    return (edge, Some(new_vertex));
                } else {
                    // Create an edge without distance
                    return (edge, None);
                }
            }
        };

        // Create the new vertex
        // This depends if there is a participating media or not
        let mut intersection_distance = its.dist;
        let (mrec, new_vertex) = if let Some(ref m) = medium {
            // Sample the participating media
            // Need to create a new ray as tfar need to store
            // the distance to the surface
            let mut ray_med = *ray;
            ray_med.tfar = intersection_distance;
            let mrec = m.sample(&ray_med, sampler.next2d());
            let new_vertex = if !mrec.exited {
                // Hit the volume
                // --- Update the distance
                intersection_distance = mrec.t;
                // --- Create the volume vertex
                let pos = Point3::from_vec(ray.o.to_vec() + ray.d * mrec.t);
                Vertex::Volume {
                    phase_function: PhaseFunction::Isotropic(),
                    pos,
                    d_in: -ray.d,
                    rr_weight: 1.0,
                    edge_in: edge,
                    edge_out: vec![],
                }
            } else {
                // Hit the surface
                Vertex::Surface {
                    its,
                    rr_weight: 1.0,
                    edge_in: edge,
                    edge_out: vec![],
                }
            };
            (Some(mrec), new_vertex)
        } else {
            (
                None,
                Vertex::Surface {
                    its,
                    rr_weight: 1.0,
                    edge_in: edge,
                    edge_out: vec![],
                },
            )
        };

        // Register the new vertex
        let new_vertex = path.register_vertex(new_vertex);

        // Update the edge information
        {
            let edge = path.edge_mut(edge);
            edge.dist = Some(intersection_distance);
            edge.vertices.1 = Some(new_vertex);
            if mrec.is_some() {
                edge.weight *= mrec.as_ref().unwrap().w;
            }
            edge.sampled_distance = mrec;
        }
        (edge, Some(new_vertex))
    }

    pub fn next_on_light_source(&self, path: &Path) -> bool {
        if let Some(v) = &self.vertices.1 {
            path.vertex(*v).on_light_source()
        } else {
            false //TODO: No env map
        }
    }

    /// Get the contribution along this edge (toward the light direction)
    /// @deprecated: This might be not optimal as it is not a recursive call.
    pub fn contribution(&self, path: &Path) -> Color {
        if let Some(v) = &self.vertices.1 {
            self.weight * self.rr_weight * path.vertex(*v).contribution(self)
        } else {
            Color::zero() //TODO: No env map
        }
    }
}

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
    }
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
            Vertex::Sensor { pos, .. } | Vertex::Light { pos, .. } | Vertex::Volume { pos, .. } => *pos
        }
    }
    pub fn on_surface(&self) -> bool {
        match *self {
            Vertex::Surface { .. } | Vertex::Light { .. } => true,
            Vertex::Sensor { .. } | Vertex::Volume { .. }  => false,
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
            Vertex::Surface { its, ..} => {
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
    pub fn have_next_vertices(&self, vertex_id: VertexID) -> bool {
        !self.next_vertices(vertex_id).is_empty()
    }
    pub fn next_vertices(&self, vertex_id: VertexID) -> Vec<(EdgeID, VertexID)> {
        let mut next_vertices = vec![];
        match self.vertex(vertex_id) {
            Vertex::Surface { edge_out, .. } 
            | Vertex::Volume { edge_out, .. } => {
                for edge_id in edge_out {
                    let edge = self.edge(*edge_id);
                    if let Some(vertex_next_id) = edge.vertices.1 {
                        next_vertices.push((*edge_id, vertex_next_id));
                    }
                }
            }
            Vertex::Sensor { edge_out, .. } 
            | Vertex::Light { edge_out, .. } => {
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
