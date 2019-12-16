use crate::bsdfs;
use crate::math::{uniform_sample_triangle, Distribution1D, Distribution1DConstruct};
use crate::structure::*;
use cgmath::*;
use std;
use tobj;

// FIXME: Support custom UV
/// Read obj file format and build a list of meshes
/// for now, only add diffuse color
/// custom texture coordinates or normals are not supported yet
pub fn load_obj(file_name: &std::path::Path) -> Result<Vec<Mesh>, tobj::LoadError> {
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
        let indices = mesh.indices
            .chunks(3)
            .map(|i| Vector3::new(i[0] as usize, i[1] as usize, i[2] as usize))
            .collect::<Vec<_>>();
        info!(" - triangles: {}", indices.len());
        let vertices = mesh.positions
            .chunks(3)
            .map(|i| Vector3::new(i[0], i[1], i[2]))
            .collect();
        // Load normal
        let normals = if mesh.normals.is_empty() {
            None
        } else {
            Some(
                mesh.normals
                    .chunks(3)
                    .map(|i| Vector3::new(i[0], i[1], i[2]))
                    .collect(),
            )
        };

        let uv = if mesh.texcoords.is_empty() {
            None
        } else {
            Some(
                mesh.texcoords
                    .chunks(2)
                    .map(|i| Vector2::new(i[0], i[1]))
                    .collect(),
            )
        };

        // Read materials and push the mesh
        let mut tri_mesh = Mesh::new(m.name, vertices, indices, normals, uv);

        // Load the BSDF informations
        tri_mesh.bsdf = {
            if let Some(id) = mesh.material_id {
                info!(" - BSDF id: {}", id);
                let mat = &materials[id];
                if !mat.diffuse_texture.is_empty() {
                    let path_texture = wk.join(&mat.diffuse_texture);
                    Box::new(bsdfs::diffuse::BSDFDiffuse {
                        diffuse: bsdfs::BSDFColor::TextureColor(bsdfs::Texture::load(
                            path_texture.to_str().unwrap(),
                        )),
                    })
                } else {
                    let diffuse_color = Color::new(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
                    Box::new(bsdfs::diffuse::BSDFDiffuse {
                        diffuse: bsdfs::BSDFColor::UniformColor(diffuse_color),
                    })
                }
            } else {
                Box::new(bsdfs::diffuse::BSDFDiffuse {
                    diffuse: bsdfs::BSDFColor::UniformColor(Color::zero()),
                })
            }
        };
        meshes.push(tri_mesh);
    }
    Ok(meshes)
}

/// (Triangle) Mesh information
pub struct Mesh {
    // Name of the triangle mesh
    pub name: String,
    // Geometrical informations
    pub vertices: Vec<Vector3<f32>>,
    pub indices: Vec<Vector3<usize>>,
    pub normals: Option<Vec<Vector3<f32>>>,
    pub uv: Option<Vec<Vector2<f32>>>,
    // Other informations
    pub bsdf: Box<dyn bsdfs::BSDF>,
    pub emission: Color,
    pub cdf: Distribution1D,
}

impl Mesh {
    pub fn new(
        name: String,
        vertices: Vec<Vector3<f32>>,
        indices: Vec<Vector3<usize>>,
        normals: Option<Vec<Vector3<f32>>>,
        uv: Option<Vec<Vector2<f32>>>,
    ) -> Mesh {
        // Construct the mesh CDF
        let mut dist_const = Distribution1DConstruct::new(indices.len());
        for id in &indices {
            let v0 = vertices[id.x];
            let v1 = vertices[id.y];
            let v2 = vertices[id.z];

            let area = (v1 - v0).cross(v2 - v0).magnitude() * 0.5;
            dist_const.add(area);
        }

        Mesh {
            name,
            vertices,
            indices,
            normals,
            uv,
            bsdf: Box::new(bsdfs::diffuse::BSDFDiffuse {
                diffuse: bsdfs::BSDFColor::UniformColor(Color::zero()),
            }),
            emission: Color::zero(),
            cdf: dist_const.normalize(),
        }
    }

    pub fn pdf(&self) -> f32 {
        1.0 / (self.cdf.normalization)
    }

    // FIXME: reuse random number
    pub fn sample(&self, s: f32, v: Point2<f32>) -> SampledPosition {
        assert!(self.normals.is_some());

        // Select a triangle
        let id = self.indices[self.cdf.sample(s)];

        let v0 = self.vertices[id.x];
        let v1 = self.vertices[id.y];
        let v2 = self.vertices[id.z];

        let normals = self.normals.as_ref().unwrap();
        let n0 = normals[id.x];
        let n1 = normals[id.y];
        let n2 = normals[id.z];

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
