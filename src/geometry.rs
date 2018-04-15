use cgmath::*;
use embree_rs;
use image;
use material::*;
use math::{uniform_sample_triangle, Distribution1D, Distribution1DConstruct};
use scene::LightSamplingPDF;
use std;
use std::sync::Arc;
use structure::Color;
use tobj;
use tools::StepRangeInt;

// FIXME: Support custom UV
/// Read obj file format and build a list of meshes
/// for now, only add diffuse color
/// custom texture coordinates or normals are not supported yet
pub fn load_obj(
    scene: &mut embree_rs::scene::SceneConstruct,
    file_name: &std::path::Path,
) -> Result<Vec<Box<Mesh>>, tobj::LoadError> {
    println!("Try to load {:?}", file_name);
    let (models, materials) = tobj::load_obj(file_name)?;
    let wk = file_name.parent().unwrap();
    info!("Working directory for loading the scene: {:?}", wk);

    // Read models
    let mut meshes = vec![];
    for m in models {
        info!("Loading model {}", m.name);
        let mesh = m.mesh;
        // Load vertex position
        info!(" - triangles: {}", mesh.indices.len() / 3);
        let vertices = mesh.positions
            .chunks(3)
            .map(|i| Vector3::new(i[0], i[1], i[2]))
            .collect();
        // Load normal
        if mesh.normals.is_empty() {
            // Normally, we want to generate the normal on the fly.
            // However, this can be difficult and as the rendering engine
            // does not support two sided BSDF, this can create problems for the user.
            // For now, just panic if the normal are not provided inside the OBJ.
            panic!("No normal provided, quit");
        }
        let normals = mesh.normals
            .chunks(3)
            .map(|i| Vector3::new(i[0], i[1], i[2]))
            .collect();

        let uv = if mesh.texcoords.is_empty() {
            vec![]
        } else {
            mesh.texcoords
                .chunks(2)
                .map(|i| Vector2::new(i[0], i[1]))
                .collect()
        };

        let trimesh = scene.add_triangle_mesh(
            embree_rs::scene::GeometryFlags::Static,
            vertices,
            normals,
            uv,
            mesh.indices,
        );
        // Read materials and push the mesh
        if let Some(id) = mesh.material_id {
            info!(" - BSDF id: {}", id);
            let mat = &materials[id];
            if !mat.diffuse_texture.is_empty() {
                let path_texture = wk.join(&mat.diffuse_texture);
                info!("Read texture: {:?}", path_texture);
                let img = image::open(path_texture).expect("Impossible to load the image");
                meshes.push(Box::new(Mesh::new(
                    m.name,
                    trimesh,
                    Box::new(BSDFDiffuse {
                        diffuse: BSDFColor::TextureColor(Texture { img }),
                    }),
                )));
            } else {
                let diffuse_color = Color::new(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
                meshes.push(Box::new(Mesh::new(
                    m.name,
                    trimesh,
                    Box::new(BSDFDiffuse {
                        diffuse: BSDFColor::UniformColor(diffuse_color),
                    }),
                )));
            }
        } else {
            meshes.push(Box::new(Mesh::new(
                m.name,
                trimesh,
                Box::new(BSDFDiffuse {
                    diffuse: BSDFColor::UniformColor(Color::zero()),
                }),
            )));
        }
    }
    Ok(meshes)
}

/// (Triangle) Mesh information
pub struct Mesh {
    pub name: String,
    pub trimesh: Arc<embree_rs::scene::TriangleMesh>,
    pub bsdf: Box<BSDF + Send + Sync>,
    pub emission: Color,
    pub cdf: Distribution1D,
}

pub struct SampledPosition {
    pub p: Point3<f32>,
    pub n: Vector3<f32>,
    pub pdf: f32,
}

impl Mesh {
    pub fn new(
        name: String,
        trimesh: Arc<embree_rs::scene::TriangleMesh>,
        bsdf: Box<BSDF + Send + Sync>,
    ) -> Mesh {
        // Construct the mesh
        assert_eq!(trimesh.indices.len() % 3, 0);
        let nb_tri = trimesh.indices.len() / 3;
        let mut dist_const = Distribution1DConstruct::new(nb_tri);
        for i in StepRangeInt::new(0, trimesh.indices.len() as usize, 3) {
            let v0 = trimesh.vertices[trimesh.indices[i] as usize];
            let v1 = trimesh.vertices[trimesh.indices[i + 1] as usize];
            let v2 = trimesh.vertices[trimesh.indices[i + 2] as usize];

            let area = (v1 - v0).cross(v2 - v0).magnitude() * 0.5;
            dist_const.add(area);
        }

        Mesh {
            name,
            trimesh,
            bsdf,
            emission: Color::zero(),
            cdf: dist_const.normalize(),
        }
    }

    pub fn pdf(&self) -> f32 {
        1.0 / (self.cdf.normalization)
    }
    pub fn flux(&self) -> f32 {
        self.cdf.normalization * self.emission.channel_max()
    }

    // FIXME: reuse random number
    pub fn sample(&self, s: f32, v: Point2<f32>) -> SampledPosition {
        // Select a triangle
        let i = self.cdf.sample(s) * 3;
        let i0 = self.trimesh.indices[i] as usize;
        let i1 = self.trimesh.indices[i + 1] as usize;
        let i2 = self.trimesh.indices[i + 2] as usize;

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
            pdf: 1.0 / (self.cdf.normalization),
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
