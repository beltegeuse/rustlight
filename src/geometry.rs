use std;
use cgmath::*;
use embree;
use std::sync::Arc;
use tobj;

// my includes
use structure::{Color,Ray};
use math::{Distribution1D, Distribution1DConstruct, uniform_sample_triangle};
use tools::StepRangeInt;

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
        // Load vertex position
        println!("{} has {} triangles", m.name, mesh.indices.len() / 3);
        let verts = mesh.positions.chunks(3).map(|i| Vector3::new(i[0], i[1], i[2])).collect();
        // Load normal
        if mesh.normals.is_empty() {
            // This is difficult to do...
            // So raise an error for now
            panic!("No normal provided, quit");
            return Err(tobj::LoadError::NormalParseError);
        }
        let normals = mesh.normals.chunks(3).map(|i| Vector3::new(i[0], i[1], i[2])).collect();

        let uv;
        if mesh.texcoords.is_empty() {
            uv = vec![];
        } else {
            uv = mesh.texcoords.chunks(2).map(|i| Vector2::new(i[0], i[1])).collect();
        }

        let trimesh = scene.new_triangle_mesh(embree::rtcore::GeometryFlags::Static,
                                              verts,
                                              normals,
                                              uv,
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
        meshes.push(Mesh::new(m.name,
            trimesh,
            diffuse_color));
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
    pub cdf : Distribution1D,
}

impl Mesh {
    pub fn new(name: String, trimesh : Arc<embree::rtcore::TriangleMesh>, bsdf: Color) -> Mesh {
        // Construct the mesh
        assert!(trimesh.indices.len() % 3 == 0);
        let nb_tri = trimesh.indices.len() / 3;
        let mut dist_const = Distribution1DConstruct::new(nb_tri);
        for i in StepRangeInt::new(0, trimesh.indices.len() as usize, 3) {
            let v0 = trimesh.vertices[trimesh.indices[i] as usize];
            let v1 = trimesh.vertices[trimesh.indices[i+1] as usize];
            let v2 = trimesh.vertices[trimesh.indices[i+2] as usize];

            let area = (v0 - v1).dot(v0 - v2).abs() * 0.5;
            dist_const.add(area);
        }

        Mesh {
            name,
            trimesh,
            bsdf,
            emission: Color::one(0.0),
            cdf : dist_const.normalize(),
        }
    }

    pub fn pdf(&self) -> f32 {
        1.0 / ( self.cdf.normalization)
    }

    // FIXME: reuse random number
    // FIXME: need to test
    pub fn sample(&self, s: f32, v: (f32,f32)) -> (Point3<f32>, Vector3<f32>, f32) {
        // Select a triangle
        let i = self.cdf.sample(s) * 3;

        let v0 = self.trimesh.vertices[self.trimesh.indices[i] as usize];
        let v1 = self.trimesh.vertices[self.trimesh.indices[i+1] as usize];
        let v2 = self.trimesh.vertices[self.trimesh.indices[i+2] as usize];

        let n0 = self.trimesh.normals[self.trimesh.indices[i] as usize];
        let n1 = self.trimesh.normals[self.trimesh.indices[i+1] as usize];
        let n2 = self.trimesh.normals[self.trimesh.indices[i+2] as usize];


        // Select barycentric coordinate on a triangle
        let b = uniform_sample_triangle(v);

        // interpol the point
        let p = v0 * b[0] + v1 * b[1] + v2 * (1.0 as f32 - b[0] - b[1]);
        let n = n0 * b[0] + n1 * b[1] + n2 * (1.0 as f32 - b[0] - b[1]);
        (Point3::new(p.x,p.y,p.z), n, 1.0 / ( std::f32::consts::PI * self.cdf.normalization))
    }

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