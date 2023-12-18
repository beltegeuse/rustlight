use crate::bsdfs;
use crate::constants::EPSILON;
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
    info!("Try to load {:?}", file_name);
    let mut options = tobj::LoadOptions::default();
    options.triangulate = true;
    let (models, materials) = tobj::load_obj(file_name, &options)?;
    let materials = materials?;
    let wk = file_name.parent().unwrap();
    info!("Working directory for loading the scene: {:?}", wk);

    // Read models
    let mut meshes = vec![];
    for m in models {
        info!("Loading model {}", m.name);
        let mesh = m.mesh;
        // Load vertex position
        let indices = mesh
            .indices
            .chunks_exact(3)
            .map(|i| Vector3::new(i[0] as usize, i[1] as usize, i[2] as usize))
            .collect::<Vec<_>>();
        info!(" - triangles: {}", indices.len());
        let vertices = mesh
            .positions
            .chunks_exact(3)
            .map(|i| Vector3::new(i[0], i[1], i[2]))
            .collect();
        // Load normal
        let normals = if mesh.normals.is_empty() {
            None
        } else {
            Some(
                mesh.normals
                    .chunks_exact(3)
                    .map(|i| Vector3::new(i[0], i[1], i[2]))
                    .collect(),
            )
        };

        let uv = if mesh.texcoords.is_empty() {
            None
        } else {
            Some(
                mesh.texcoords
                    .chunks_exact(2)
                    .map(|i| Vector2::new(i[0], i[1]))
                    .collect(),
            )
        };

        // Read materials and push the mesh
        let mut tri_mesh = Mesh::new(m.name, vertices, indices, normals, uv).unwrap();

        // Load the BSDF informations
        tri_mesh.bsdf = {
            if let Some(id) = mesh.material_id {
                info!(" - BSDF id: {}", id);
                let mat = &materials[id];
                if let Some(tx) = &mat.diffuse_texture {
                    let path_texture = wk.join(&tx);
                    Box::new(bsdfs::diffuse::BSDFDiffuse {
                        diffuse: bsdfs::BSDFColor::Bitmap {
                            img: Bitmap::read(&path_texture.to_str().unwrap()),
                        },
                    })
                } else if let Some(kd) = mat.diffuse {
                    let diffuse_color = Color::new(kd[0], kd[1], kd[2]);
                    Box::new(bsdfs::diffuse::BSDFDiffuse {
                        diffuse: bsdfs::BSDFColor::Constant(diffuse_color),
                    })
                } else {
                    // Default?
                    Box::new(bsdfs::diffuse::BSDFDiffuse {
                        diffuse: bsdfs::BSDFColor::Constant(Color::zero()),
                    })
                }
            } else {
                Box::new(bsdfs::diffuse::BSDFDiffuse {
                    diffuse: bsdfs::BSDFColor::Constant(Color::zero()),
                })
            }
        };
        meshes.push(tri_mesh);
    }
    Ok(meshes)
}

pub enum EmissionType {
    Zero,
    Color { v: Color },
    HSV { scale: f32 },
    Texture { scale: f32, img: Bitmap },
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
    pub emission: EmissionType,
    pub cdf: Option<Distribution1D>,
}

impl Mesh {
    pub fn new(
        name: String,
        vertices: Vec<Vector3<f32>>,
        indices: Vec<Vector3<usize>>,
        mut normals: Option<Vec<Vector3<f32>>>,
        uv: Option<Vec<Vector2<f32>>>,
    ) -> Option<Mesh> {
        // Construct the mesh CDF
        let mut dist_const = Distribution1DConstruct::new(indices.len());
        for id in &indices {
            let v0 = vertices[id.x];
            let v1 = vertices[id.y];
            let v2 = vertices[id.z];

            let area = (v1 - v0).cross(v2 - v0).magnitude() * 0.5;
            dist_const.add(area);
        }

        // Normalize all the normal if if it is necessary
        // Indeed, sometimes the normal are not properly normalized
        let mut nb_wrong_normals = 0;
        if let Some(ref mut ns) = normals.as_mut() {
            for n in ns.iter_mut() {
                let l = n.dot(*n);
                if l == 0.0 {
                    nb_wrong_normals += 1;
                    // TODO: The problem is n_s that is incorrect...
                    // There is not much we can do
                } else if l != 1.0 {
                    *n /= l.sqrt();
                }
            }
        }
        if normals.is_some() {
            let nb_normal = normals.as_ref().unwrap().len();
            if nb_wrong_normals > 0 {
                warn!("Detected wrong normal: {}/{}", nb_wrong_normals, nb_normal);
                if nb_normal == nb_wrong_normals {
                    error!("All normal are wrong, we will delete the normal informations");
                    normals = None;
                }
            }
        }
        if dist_const.elements.is_empty() {
            warn!("Empty meshs, abording the creating of this mesh");
            None
        } else {
            Some(Mesh {
                name,
                vertices,
                indices,
                normals,
                uv,
                bsdf: Box::new(bsdfs::diffuse::BSDFDiffuse {
                    diffuse: bsdfs::BSDFColor::Constant(Color::zero()),
                }),
                emission: EmissionType::Zero,
                cdf: Some(dist_const.normalize()),
            })
        }
    }

    pub fn emit(&self, uv: &Option<Vector2<f32>>) -> Color {
        match &self.emission {
            EmissionType::Zero => Color::zero(),
            EmissionType::Color { v } => *v,
            EmissionType::HSV { scale } => {
                let c1 = Color {
                    r: 1.0,
                    g: 0.0,
                    b: 0.0,
                };
                let c2 = Color {
                    r: 0.0,
                    g: 1.0,
                    b: 0.0,
                };
                let uv = uv.unwrap();
                let x = uv.x.abs() % 1.0;
                let c = x * c1 + (1.0 - x) * c2;
                c * (*scale)
            }
            EmissionType::Texture { scale, img } => img.pixel_uv(uv.unwrap()) * (*scale),
        }
    }

    pub fn build_cdf(&mut self) {
        warn!("Build CDF have been already done.");
        //assert!(self.cdf.is_none());
        let mut dist_const = Distribution1DConstruct::new(self.indices.len());
        for id in &self.indices {
            let v0 = self.vertices[id.x];
            let v1 = self.vertices[id.y];
            let v2 = self.vertices[id.z];

            let area = (v1 - v0).cross(v2 - v0).magnitude() * 0.5;
            dist_const.add(area);
        }
        self.cdf = Some(dist_const.normalize());
    }

    pub fn pdf(&self) -> f32 {
        1.0 / (self.cdf.as_ref().unwrap().total())
    }
    pub fn pdf_tri(&self, primitive_id: usize) -> f32 {
        let id = self.indices[primitive_id];
        let v0 = self.vertices[id.x];
        let v1 = self.vertices[id.y];
        let v2 = self.vertices[id.z];

        let area_tri = (v1 - v0).cross(v2 - v0).magnitude() * 0.5;
        1.0 / area_tri
    }

    pub fn naive_intersection(&self, ray: &Ray) -> Option<Intersection> {
        let mut its = IntersectionUV {
            t: std::f32::MAX,
            p: Point3::new(0.0, 0.0, 0.0),
            n: Vector3::new(0.0, 0.0, 0.0),
            u: 0.0,
            v: 0.0,
        };

        let mut id_t = 0;
        for i in 0..self.indices.len() {
            if self.intersection_tri(i, &ray.o, &ray.d, &mut its) {
                id_t = i;
            }
        }

        if its.t == std::f32::MAX {
            None
        } else {
            Some(Intersection::fill_intersection(
                self, id_t, its.u, its.v, ray, its.n, its.t, its.p,
            ))
        }
    }

    pub fn sample_tri(&self, primitive_id: usize, v: Point2<f32>) -> SampledPosition {
        let id = self.indices[primitive_id];
        let v0 = self.vertices[id.x];
        let v1 = self.vertices[id.y];
        let v2 = self.vertices[id.z];

        // Select barycentric coordinate on a triangle
        let b = uniform_sample_triangle(v);

        // Geometry normals
        let pos = v0 * b[0] + v1 * b[1] + v2 * (1.0 as f32 - b[0] - b[1]);
        let n_g = {
            let u = v1 - v0;
            let v = v2 - v0;
            v.cross(u).normalize()
        };

        // Correct geometry normal if exist
        let n_g = match &self.normals {
            Some(normals) => {
                let n0 = normals[id.x];
                let n1 = normals[id.y];
                let n2 = normals[id.z];
                // Interpolate normal
                let n = n0 * b[0] + n1 * b[1] + n2 * (1.0 as f32 - b[0] - b[1]);
                let n_l = n.magnitude2();

                // Normalize the shading normal
                let n = if n_l == 0.0 {
                    n_g
                } else if n_l != 1.0 {
                    n / n_l.sqrt()
                } else {
                    n
                };

                // TODO: In the cornel-box, geometry normal seems not correct
                //  Need to continue to check this case.

                // Make shading normal facing geometric one
                // if n_g.dot(n) < 0.0 {
                //     -n
                // } else {
                //     n
                // }

                // Only gives the n_g for the moment
                if n_g.dot(n) < 0.0 {
                    -n_g
                } else {
                    n_g
                }
            }
            None => n_g,
        };

        let uv = match &self.uv {
            Some(uv) => {
                let n0 = uv[id.x];
                let n1 = uv[id.y];
                let n2 = uv[id.z];
                let uv_0 = (n0 * b[0] + n1 * b[1] + n2 * (1.0 as f32 - b[0] - b[1])).normalize();
                Some(uv_0)
            }
            None => None,
        };

        let area_tri = (v1 - v0).cross(v2 - v0).magnitude() * 0.5;

        SampledPosition {
            p: Point3::from_vec(pos),
            n: n_g,
            uv,
            pdf: PDF::Area(1.0 / area_tri),
            primitive_id: Some(primitive_id),
        }
    }

    // FIXME: reuse random number
    pub fn sample(&self, s: f32, v: Point2<f32>) -> SampledPosition {
        // Select a triangle
        let primitive_id = self.cdf.as_ref().unwrap().sample_discrete(s);
        // Sample a point on the triangle
        let mut res = self.sample_tri(primitive_id, v);
        // Modify the pdf to show we pick triangle prop to their area
        res.pdf = PDF::Area(1.0 / self.cdf.as_ref().unwrap().total());
        res
    }

    // Triangle methods
    pub fn middle_tri(&self, i: usize) -> Vector3<f32> {
        let id = self.indices[i];
        let v0 = self.vertices[id.x];
        let v1 = self.vertices[id.y];
        let v2 = self.vertices[id.z];
        (v0 + v1 + v2) / 3.0
    }
    pub fn intersection_tri(
        &self,
        i: usize,
        p_c: &Point3<f32>,
        d_c: &Vector3<f32>,
        its: &mut IntersectionUV,
    ) -> bool {
        let id = self.indices[i];
        let v0 = self.vertices[id.x];
        let v1 = self.vertices[id.y];
        let v2 = self.vertices[id.z];

        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let n_geo = e1.cross(e2).normalize();
        let denom = d_c.dot(n_geo);
        if denom == 0.0 {
            return false;
        }
        // Distance for intersection
        let t = -(p_c - v0).dot(n_geo) / denom;
        if t < 0.0 {
            return false;
        }
        let p = p_c + t * d_c;
        let det = e1.cross(e2).magnitude();
        let u0 = e1.cross(p.to_vec() - v0);
        let v0 = (p.to_vec() - v0).cross(e2);
        if u0.dot(n_geo) < 0.0 || v0.dot(n_geo) < 0.0 {
            return false;
        }
        let v = u0.magnitude() / det;
        let u = v0.magnitude() / det;
        if u < 0.0 || v < 0.0 || u > 1.0 || v > 1.0 {
            return false;
        }
        if u + v <= 1.0 {
            // TODO: Review the condition because
            //      for now it only return true
            //      if the itersection is updated
            if t < its.t && t > 0.00001 {
                // Avoid self intersection
                // FIXME: Make this code cleaner
                its.t = t;
                its.u = u;
                its.v = v;
                its.p = p;
                its.n = n_geo;
                return true;
            }
        }
        false
    }

    pub fn is_light(&self) -> bool {
        match &self.emission {
            EmissionType::Zero => false,
            _ => true,
        }
    }

    pub fn discard_normals(&mut self) {
        self.normals = None;
    }

    pub fn compute_aabb_tri(&self, i: usize) -> AABB {
        let id = self.indices[i];
        let mut aabb = AABB::default();
        for s in 0..3 {
            aabb = aabb.union_vec(&self.vertices[id[s]]);
        }

        // Make sure the AABB to be non degenerative
        let s = aabb.size();
        for i in 0..3 {
            if s[i] < EPSILON {
                aabb.p_max[i] += EPSILON;
                aabb.p_min[i] -= EPSILON;
            }
        }
        aabb
    }

    pub fn compute_aabb(&self) -> AABB {
        let mut aabb = AABB::default();
        for v in &self.vertices {
            aabb = aabb.union_vec(v)
        }

        // Make sure the AABB to be non degenerative
        let s = aabb.size();
        for i in 0..3 {
            if s[i] < EPSILON {
                aabb.p_max[i] += EPSILON;
                aabb.p_min[i] -= EPSILON;
            }
        }

        aabb
    }
}
