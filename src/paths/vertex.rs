use cgmath::*;
use geometry::Mesh;
use scene::*;
use std::cell::RefCell;
use std::rc::{Rc, Weak};
use std::sync::Arc;
use structure::*;

#[derive(Clone)]
pub struct Edge<'a> {
    /// Geometric informations
    pub dist: Option<f32>,
    pub d: Vector3<f32>,
    /// Connecting two vertices
    pub vertices: (Weak<VertexPtr<'a>>, Option<Rc<VertexPtr<'a>>>),
    /// Sampling information
    pub pdf_distance: f32,
    pub pdf_direction: PDF,
    pub weight: Color,
    pub rr_weight: f32,
    pub id_sampling: usize,
}

impl<'a> Edge<'a> {
    pub fn from_vertex(
        org_vertex: &Rc<VertexPtr<'a>>,
        pdf_direction: PDF,
        weight: Color,
        rr_weight: f32,
        next_vertex: &Rc<VertexPtr<'a>>,
        id_sampling: usize,
    ) -> Rc<EdgePtr<'a>> {
        let mut d = next_vertex.borrow().position() - org_vertex.borrow().position();
        let dist = d.magnitude();
        d /= dist;

        let edge = Rc::new(RefCell::new(Edge {
            dist: Some(dist),
            d,
            vertices: (Rc::downgrade(&org_vertex), Some(next_vertex.clone())),
            pdf_distance: 1.0,
            pdf_direction,
            weight,
            rr_weight,
            id_sampling,
        }));

        match *next_vertex.borrow_mut() {
            Vertex::Emitter(ref mut v) => v.edge_in = Some(Rc::downgrade(&edge)),
            _ => unimplemented!(),
        };

        edge
    }

    pub fn from_ray(
        ray: Ray,
        org_vertex: &Rc<VertexPtr<'a>>,
        pdf_direction: PDF,
        weight: Color,
        rr_weight: f32,
        scene: &'a Scene,
        id_sampling: usize,
    ) -> (Rc<EdgePtr<'a>>, Option<Rc<VertexPtr<'a>>>) {
        // TODO: When there will be volume, we need to sample a distance inside the volume
        let edge = Rc::new(RefCell::new(Edge {
            dist: None,
            d: ray.d,
            vertices: (Rc::downgrade(org_vertex), None),
            pdf_distance: 1.0,
            pdf_direction,
            weight,
            rr_weight,
            id_sampling,
        }));

        let its = match scene.trace(&ray) {
            Some(its) => its,
            None => {
                // Create an edge without distance
                return (edge, None);
            }
        };

        // Create the new vertex
        let intersection_distance = its.dist;
        let new_vertex = Rc::new(RefCell::new(Vertex::Surface(SurfaceVertex {
            its,
            rr_weight: 1.0,
            edge_in: Rc::downgrade(&edge),
            edge_out: vec![],
        })));

        // Update the edge information
        {
            let mut edge = edge.borrow_mut();
            edge.dist = Some(intersection_distance);
            edge.vertices.1 = Some(new_vertex.clone());
        }
        (edge, Some(new_vertex))
    }

    pub fn next_on_light_source(&self) -> bool {
        if let Some(v) = &self.vertices.1 {
            return v.borrow().on_light_source();
        } else {
            return false; //TODO: No env map
        }
    }

    /// Get the contribution along this edge (toward the light direction)
    /// @deprecated: This might be not optimal as it is not a recursive call.
    pub fn contribution(&self) -> Color {
        if let Some(v) = &self.vertices.1 {
            return self.weight * self.rr_weight * v.borrow().contribution(self);
        } else {
            return Color::zero(); //TODO: No env map
        }
    }
}

#[derive(Clone)]
pub struct SensorVertex<'a> {
    pub uv: Point2<f32>,
    pub pos: Point3<f32>,
    pub edge_in: Option<Weak<EdgePtr<'a>>>,
    pub edge_out: Option<Rc<EdgePtr<'a>>>,
}

#[derive(Clone)]
pub struct SurfaceVertex<'a> {
    pub its: Intersection<'a>,
    pub rr_weight: f32,
    pub edge_in: Weak<EdgePtr<'a>>,
    pub edge_out: Vec<Rc<EdgePtr<'a>>>,
}

#[derive(Clone)]
pub struct EmitterVertex<'a> {
    pub pos: Point3<f32>,
    pub n: Vector3<f32>,
    pub mesh: &'a Arc<Mesh>,
    pub edge_in: Option<Weak<EdgePtr<'a>>>,
    pub edge_out: Option<Rc<EdgePtr<'a>>>,
}

#[derive(Clone)]
pub enum Vertex<'a> {
    Sensor(SensorVertex<'a>),
    Surface(SurfaceVertex<'a>),
    Emitter(EmitterVertex<'a>),
}
impl<'a> Vertex<'a> {
    pub fn position(&self) -> Point3<f32> {
        match *self {
            Vertex::Surface(ref v) => v.its.p,
            Vertex::Sensor(ref v) => v.pos,
            Vertex::Emitter(ref v) => v.pos,
        }
    }

    pub fn on_light_source(&self) -> bool {
        match *self {
            Vertex::Surface(ref v) => !v.its.mesh.emission.is_zero(),
            Vertex::Sensor(ref _v) => false,
            Vertex::Emitter(ref _v) => true,
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
            Vertex::Emitter(ref v) => v.mesh.emission, // FIXME: Check the normal orientation
        }
    }
}

pub type VertexPtr<'a> = RefCell<Vertex<'a>>;
pub type EdgePtr<'a> = RefCell<Edge<'a>>;
