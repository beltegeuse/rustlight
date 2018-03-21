use std;
use cgmath::*;
use embree_rs;
use std::sync::Arc;
use tobj;

// my includes
use structure::{Color};
use math::{Distribution1D, Distribution1DConstruct, uniform_sample_triangle};
use tools::StepRangeInt;
use material::{BSDF,BSDFDiffuse};
use structure::Ray;
use scene::LightSamplingPDF;

// FIXME: Support custom UV
// FIXME: Support custom normal
/// Read obj file format and build a list of meshes
/// for now, only add diffuse color
/// custom texcoords or normals are not supported yet
pub fn load_obj(scene: &mut embree_rs::scene::Scene, file_name: & std::path::Path) -> Result<Vec<Box<Mesh>>, tobj::LoadError> {
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
            //return Err(tobj::LoadError::NormalParseError);
        }
        let normals = mesh.normals.chunks(3).map(|i| Vector3::new(i[0], i[1], i[2])).collect();

        let uv = if mesh.texcoords.is_empty() { vec![] }
            else { mesh.texcoords.chunks(2).map(|i| Vector2::new(i[0], i[1])).collect() };

        let trimesh = scene.new_triangle_mesh(embree_rs::scene::GeometryFlags::Static,
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
            diffuse_color = Color::zero();
        }

        // Add the mesh info
        meshes.push(Box::new(Mesh::new(m.name,
            trimesh,
            diffuse_color)));
    }
    Ok(meshes)
}

// FIXME: add distribution 1D to sample a point on triangles
/// (Triangle) Mesh information
pub struct Mesh {
    pub name : String,
    pub trimesh : Arc<embree_rs::scene::TriangleMesh>,
    pub bsdf : Box<BSDF + Send + Sync>,
    pub emission : Color,
    pub cdf : Distribution1D,
}

pub struct SampledPosition {
    pub p: Point3<f32>,
    pub n: Vector3<f32>,
    pub pdf: f32,
}

impl Mesh {
    pub fn new(name: String, trimesh : Arc<embree_rs::scene::TriangleMesh>, diffuse: Color) -> Mesh {
        // Construct the mesh
        assert!(trimesh.indices.len() % 3 == 0);
        let nb_tri = trimesh.indices.len() / 3;
        let mut dist_const = Distribution1DConstruct::new(nb_tri);
        for i in StepRangeInt::new(0, trimesh.indices.len() as usize, 3) {
            let v0 = trimesh.vertices[trimesh.indices[i] as usize];
            let v1 = trimesh.vertices[trimesh.indices[i+1] as usize];
            let v2 = trimesh.vertices[trimesh.indices[i+2] as usize];

            let area = (v1 - v0).cross(v2 - v0).magnitude() * 0.5;
            dist_const.add(area);
        }

        Mesh {
            name,
            trimesh,
            bsdf : Box::new(BSDFDiffuse { diffuse }),
            emission: Color::zero(),
            cdf : dist_const.normalize(),
        }
    }

    pub fn pdf(&self) -> f32 {
        1.0 / ( self.cdf.normalization)
    }
    pub fn flux(&self) -> f32 { self.cdf.normalization * self.emission.channel_max()}

    // FIXME: reuse random number
    pub fn sample(&self, s: f32, v: Point2<f32>) -> SampledPosition {
        // Select a triangle
        let i = self.cdf.sample(s) * 3;
        let i0 = self.trimesh.indices[i] as usize;
        let i1 = self.trimesh.indices[i+1] as usize;
        let i2 = self.trimesh.indices[i+2] as usize;

        let v0 = self.trimesh.vertices[i0];
        let v1 = self.trimesh.vertices[i1];
        let v2 = self.trimesh.vertices[i2];

        let n0 = self.trimesh.normals[i0];
        let n1 = self.trimesh.normals[i1];
        let n2 = self.trimesh.normals[i2];

        // Select barycentric coordinate on a triangle
        let b = uniform_sample_triangle(v);

        // interpol the point
        let pos = v0 * b[0] + v1 * b[1] + v2 * (1.0 as f32 - b[0] - b[1]);
        let normal = n0 * b[0] + n1 * b[1] + n2 * (1.0 as f32 - b[0] - b[1]);
        SampledPosition {
            p: Point3::from_vec(pos),
            n: normal,
            pdf: 1.0 / ( self.cdf.normalization),
        }
    }

    pub fn is_light(&self) -> bool {
        !self.emission.is_zero()
    }

    /// PDF value when we intersect the light
    pub fn direct_pdf(&self, light_sampling: LightSamplingPDF) -> f32 {
        let cos_light = light_sampling.n.dot(-light_sampling.dir).max(0.0);
        if cos_light == 0.0 {
            0.0
        } else {
            let geom_inv = (light_sampling.p - light_sampling.o).magnitude2() / cos_light;
            self.pdf() * geom_inv
        }
    }
}