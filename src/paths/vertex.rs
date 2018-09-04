use bsdfs::*;
use cgmath::*;
use samplers::*;
use scene::*;
use structure::*;
use geometry::Mesh;
use std::sync::Arc;
use std::rc::{Rc,Weak};
use std::cell::RefCell;
use Scale;

#[derive(Clone)]
pub struct Edge<'a> {
    /// Geometric informations
    pub dist: Option<f32>,
    pub d: Vector3<f32>,
    /// Connecting two vertices
    pub vertices: (Weak<VertexPtr<'a>>,
            Option<Rc<VertexPtr<'a>>>),
    /// Sampling information
    pub pdf_distance: f32,
    pub pdf_direction: PDF,
    pub weight: Color,
    pub rr_weight: f32,
}


impl<'a> Edge<'a> {
    pub fn from_vertex(org_vertex: &Rc<VertexPtr<'a>>,
        pdf_direction: PDF,
        weight: Color,
        rr_weight: f32,
        next_vertex: &Rc<VertexPtr<'a>>,
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
            weight, rr_weight}));
        
        match *next_vertex.borrow_mut() {
            Vertex::Emitter(ref mut v) => { v.edge = Some(Rc::downgrade(&edge)) },
            _ => unimplemented!(), 
        };

        return edge;
    }

    pub fn from_ray(ray: Ray, 
        org_vertex: &Rc<VertexPtr<'a>>, 
        pdf_direction: PDF, 
        weight: Color, 
        rr_weight: f32,
        scene: &'a Scene) -> (Rc<EdgePtr<'a>>, Option<Rc<VertexPtr<'a>>>) { 
        // TODO: When there will be volume, we need to sample a distance inside the volume
        let edge = Rc::new(RefCell::new(Edge {  
            dist: None, 
            d: ray.d, 
            vertices: (Rc::downgrade(org_vertex), None), 
            pdf_distance: 1.0, 
            pdf_direction, 
            weight, rr_weight}));
        
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
                its: its,
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
        (
            edge,
            Some(new_vertex),
        )
    }
}

#[derive(Clone)]
pub struct SensorVertex<'a> {
    pub uv: Point2<f32>,
    pub pos: Point3<f32>,
    pub edge: Option<Rc<EdgePtr<'a>>>,
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
    pub mesh: &'a Mesh,
    pub edge: Option<Weak<EdgePtr<'a>>>,
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
}

pub type VertexPtr<'a> = RefCell<Vertex<'a>>;
pub type EdgePtr<'a> = RefCell<Edge<'a>>;