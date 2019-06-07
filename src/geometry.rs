use crate::bsdfs;
use crate::math::{uniform_sample_triangle, Distribution1D, Distribution1DConstruct};
use crate::structure::*;
use crate::tools::StepRangeInt;
use cgmath::*;
use embree_rs;
use std;
use std::sync::Arc;
use tobj;

// FIXME: Support custom UV
/// Read obj file format and build a list of meshes
/// for now, only add diffuse color
/// custom texture coordinates or normals are not supported yet
pub fn load_obj(
    device: &embree_rs::Device,
    scene: &mut embree_rs::SceneConstruct,
    file_name: &std::path::Path,
) -> Result<Vec<Mesh>, tobj::LoadError> {
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
            .map(|i| Point3::new(i[0], i[1], i[2]))
            .collect();
        // Load normal
        let normals = if mesh.normals.is_empty() {
            // Only rely on face normals
            Vec::new()
        } else {
            mesh.normals
                .chunks(3)
                .map(|i| Vector3::new(i[0], i[1], i[2]))
                .collect()
        };

        let uv = if mesh.texcoords.is_empty() {
            vec![]
        } else {
            mesh.texcoords
                .chunks(2)
                .map(|i| Vector2::new(i[0], i[1]))
                .collect()
        };

        let trimesh = scene.add_triangle_mesh(device, vertices, normals, uv, mesh.indices);
        // Read materials and push the mesh
        if let Some(id) = mesh.material_id {
            info!(" - BSDF id: {}", id);
            let mat = &materials[id];
            if !mat.diffuse_texture.is_empty() {
                let path_texture = wk.join(&mat.diffuse_texture);
                meshes.push(Mesh::new(
                    m.name,
                    trimesh,
                    Box::new(bsdfs::diffuse::BSDFDiffuse {
                        diffuse: bsdfs::BSDFColor::TextureColor(bsdfs::Texture::load(
                            path_texture.to_str().unwrap(),
                        )),
                    }),
                ));
            } else {
                let diffuse_color = Color::new(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
                meshes.push(Mesh::new(
                    m.name,
                    trimesh,
                    Box::new(bsdfs::diffuse::BSDFDiffuse {
                        diffuse: bsdfs::BSDFColor::UniformColor(diffuse_color),
                    }),
                ));
            }
        } else {
            meshes.push(Mesh::new(
                m.name,
                trimesh,
                Box::new(bsdfs::diffuse::BSDFDiffuse {
                    diffuse: bsdfs::BSDFColor::UniformColor(Color::zero()),
                }),
            ));
        }
    }
    Ok(meshes)
}

pub struct AABB {
    pub min: Vector3<f32>,
    pub max: Vector3<f32>,
}
impl Default for AABB {
    fn default() -> Self {
        Self {
            min: Vector3::new(std::f32::MAX, std::f32::MAX, std::f32::MAX),
            max: Vector3::new(std::f32::MIN, std::f32::MIN, std::f32::MIN),
        }
    }
}
impl AABB {
    pub fn center(&self) -> Vector3<f32> {
        (self.min + self.max) * 0.5
    }
    pub fn surface_area(&self) -> f32 {
        let d = self.max - self.min;
        2.0 * (d.x * d.y + d.y + d.z + d.z * d.x)
    }
    pub fn merge_point(self, p: Vector3<f32>) -> AABB {
        AABB {
            min: Vector3::new(
                self.min.x.min(p.x),
                self.min.y.min(p.y),
                self.min.z.min(p.z),
            ),
            max: Vector3::new(
                self.max.x.max(p.x),
                self.max.y.max(p.y),
                self.max.z.max(p.z),
            ),
        }
    }
    pub fn merge_aabb(self, b: AABB) -> AABB {
        AABB {
            min: Vector3::new(
                self.min.x.min(b.min.x),
                self.min.y.min(b.min.y),
                self.min.z.min(b.min.z),
            ),
            max: Vector3::new(
                self.max.x.max(b.max.x),
                self.max.y.max(b.max.y),
                self.max.z.max(b.max.z),
            ),
        }
    }
}

pub struct Triangle {
    pub p1: Vector3<f32>,
    pub e1: Vector3<f32>,
    pub e2: Vector3<f32>,
    pub aabb: AABB,
}

/// (Triangle) Mesh information
pub struct Mesh {
    pub name: String,
    pub trimesh: Arc<embree_rs::TriangleMesh>,
    pub bsdf: Box<bsdfs::BSDF>,
    pub emission: Color,
    pub cdf: Distribution1D,
}

impl Mesh {
    pub fn new(
        name: String,
        trimesh: Arc<embree_rs::TriangleMesh>,
        bsdf: Box<bsdfs::BSDF>,
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

    // FIXME: reuse random number
    pub fn sample(&self, s: f32, v: Point2<f32>) -> SampledPosition {
        // Select a triangle
        let i = self.cdf.sample(s) * 3;
        let i0 = self.trimesh.indices[i] as usize;
        let i1 = self.trimesh.indices[i + 1] as usize;
        let i2 = self.trimesh.indices[i + 2] as usize;

        let v0 = self.trimesh.vertices[i0].to_vec();
        let v1 = self.trimesh.vertices[i1].to_vec();
        let v2 = self.trimesh.vertices[i2].to_vec();

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
            pdf: PDF::Area(1.0 / (self.cdf.normalization)),
        }
    }

    pub fn is_light(&self) -> bool {
        !self.emission.is_zero()
    }
}
