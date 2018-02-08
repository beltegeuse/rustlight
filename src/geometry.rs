use std;
use cgmath::*;
use embree;
use std::sync::Arc;
use tobj;

// my includes
use structure::{Color,Ray};

// FIXME: Support custom UV
// FIXME: Support custom normal
/// Read obj file format and build a list of meshes
/// for now, only add diffuse color
/// custom texcoords or normals are not supported yet
pub fn load_obj(scene: &mut embree::rtcore::Scene, file_name: & std::path::Path) -> Result<Vec<Mesh>, tobj::LoadError> {
    println!("Try to load {:?}", file_name);
    let (models, materials) = tobj::load_obj(file_name)?;

    // Read models
    let mut meshes = vec![];
    for m in models {
        println!("Loading model {}", m.name);
        let mesh = m.mesh;
        println!("{} has {} triangles", m.name, mesh.indices.len() / 3);
        let verts = mesh.positions.chunks(3).map(|i| Vector4::new(i[0], i[1], i[2], 1.0)).collect();
        let trimesh = scene.new_triangle_mesh(embree::rtcore::GeometryFlags::Static,
                                              verts,
                                              mesh.indices);
        // Read materials
        let diffuse_color;
        if let Some(id) = mesh.material_id {
            println!("found bsdf id: {}", id);
            let mat = &materials[id];
            diffuse_color = Color::new(mat.diffuse[0],
                                       mat.diffuse[1],
                                       mat.diffuse[2]);
        } else {
            diffuse_color = Color::one(0.0);
        }

        // Add the mesh info
        meshes.push(Mesh {
            name: m.name,
            trimesh,
            bsdf: diffuse_color,
            emission: Color::one(0.0),
        })
    }
    Ok(meshes)
}

// FIXME: add distribution 1D to sample a point on triangles
/// (Triangle) Mesh information
pub struct Mesh {
    pub name : String,
    pub trimesh : Arc<embree::rtcore::TriangleMesh>,
    pub bsdf : Color, // FIXME: Only diffuse color for now
    pub emission : Color,
}

impl Mesh {
    pub fn is_light(&self) -> bool {
        return !self.emission.is_zero();
    }
}

// FIXME: See how to re-integrate this code
///////////////////////// Legacy code
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