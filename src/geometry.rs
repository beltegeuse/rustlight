use cgmath::*;
use embree;
use std::sync::Arc;

// my includes
use structure::{Color,Ray};

pub struct Mesh {
    pub name : String,
    pub trimesh : Arc<embree::rtcore::TriangleMesh>,
    pub bsdf : Color, // Diffuse color
    pub emission : Color,
}

impl Mesh {
    pub fn is_light(&self) -> bool {
        return !self.emission.is_zero();
    }
}

pub struct Sphere {
    pub pos: Point3<f32>,
    pub radius: f32,
    pub color: Color,
}

pub trait Intersectable {
    fn intersect(&self, ray: &Ray) -> Option<f32>;
}

impl Intersectable for Sphere {
    fn intersect(&self, ray: &Ray) -> Option<f32> {
        let l = self.pos - ray.o;
        let adj = l.dot(ray.d);
        let d2 = l.dot(l) - (adj * adj);
        let radius2 = self.radius * self.radius;
        if d2 > radius2 {
            return None;
        }
        let thc = (radius2 - d2).sqrt();
        let t0 = adj - thc;
        let t1 = adj + thc;

        if t0 < 0.0 && t1 < 0.0 {
            return None;
        }

        let distance = if t0 < t1 { t0 } else { t1 };
        Some(distance)
    }
}