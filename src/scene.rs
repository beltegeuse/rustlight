use crate::camera::Camera;
use crate::emitter::*;
use crate::geometry;
use crate::math::Distribution1DConstruct;

use crate::math::Frame;
use crate::structure::*;
use cgmath::*;
pub trait Acceleration: Sync + Send {
    fn trace(&self, ray: &Ray) -> Option<Intersection>;
    fn visible(&self, p0: &Point3<f32>, p1: &Point3<f32>) -> bool;
}

pub struct EmbreeAcceleration<'a, 'scene> {
    pub scene: &'a Scene,
    pub rtscene: embree_rs::CommittedScene<'scene>,
}

impl<'a, 'scene> EmbreeAcceleration<'a, 'scene> {
    pub fn new(
        scene: &'a Scene,
        embree_scene: &'scene embree_rs::Scene,
    ) -> EmbreeAcceleration<'a, 'scene> {
        EmbreeAcceleration {
            scene,
            rtscene: embree_scene.commit(),
        }
    }
}

impl<'a, 'scene> Acceleration for EmbreeAcceleration<'a, 'scene> {
    fn trace(&self, ray: &Ray) -> Option<Intersection> {
        let mut intersection_ctx = embree_rs::IntersectContext::coherent();
        let embree_ray = embree_rs::Ray::segment(
            Vector3::new(ray.o.x, ray.o.y, ray.o.z),
            ray.d,
            ray.tnear,
            ray.tfar,
        );
        let mut ray_hit = embree_rs::RayHit::new(embree_ray);
        self.rtscene.intersect(&mut intersection_ctx, &mut ray_hit);
        if ray_hit.hit.hit() {
            let mesh = &self.scene.meshes[ray_hit.hit.geomID as usize];
            let index = mesh.indices[ray_hit.hit.primID as usize];

            // Retrive the mesh
            // The geometric normal is not normalized...
            let mut n_g = Vector3::new(ray_hit.hit.Ng_x, ray_hit.hit.Ng_y, ray_hit.hit.Ng_z);
            let n_g_dot = n_g.dot(n_g);
            if n_g_dot != 1.0 {
                n_g /= n_g_dot.sqrt();
            }

            let n_s = if let Some(ref normals) = mesh.normals {
                let d0 = &normals[index.x];
                let d1 = &normals[index.y];
                let d2 = &normals[index.z];
                let mut n_s = d0 * (1.0 - ray_hit.hit.u - ray_hit.hit.v)
                    + d1 * ray_hit.hit.u
                    + d2 * ray_hit.hit.v;
                if n_g.dot(n_s) < 0.0 {
                    n_s = -n_s;
                }
                n_s
            } else {
                n_g
            };

            // TODO: Hack for now for make automatic twosided.
            let (n_s, n_g) =
                if mesh.bsdf.is_twosided() && mesh.emission.is_zero() && ray.d.dot(n_s) > 0.0 {
                    (
                        Vector3::new(-n_s.x, -n_s.y, -n_s.z),
                        Vector3::new(-n_g.x, -n_g.y, -n_g.z),
                    )
                } else {
                    (n_s, n_g)
                };


            // UV interpolation
            let uv = if let Some(ref uv_data) = mesh.uv {
                let d0 = &uv_data[index.x];
                let d1 = &uv_data[index.y];
                let d2 = &uv_data[index.z];
                Some(
                    d0 * (1.0 - ray_hit.hit.u - ray_hit.hit.v)
                        + d1 * ray_hit.hit.u
                        + d2 * ray_hit.hit.v,
                )
            } else {
                None
            };

            let frame = Frame::new(n_s);
            let wi = frame.to_local(-ray.d);
            Some(Intersection {
                dist: ray_hit.ray.tfar,
                n_g,
                n_s,
                p: Point3::new(
                    ray_hit.ray.org_x + ray_hit.ray.tfar * ray_hit.ray.dir_x,
                    ray_hit.ray.org_y + ray_hit.ray.tfar * ray_hit.ray.dir_y,
                    ray_hit.ray.org_z + ray_hit.ray.tfar * ray_hit.ray.dir_z,
                ),
                uv,
                mesh,
                frame,
                wi,
            })
        } else {
            None
        }
    }
    fn visible(&self, p0: &Point3<f32>, p1: &Point3<f32>) -> bool {
        let mut intersection_ctx = embree_rs::IntersectContext::coherent();
        let mut d = p1 - p0;
        let length = d.magnitude();
        d /= length;
        let mut embree_ray = embree_rs::Ray::segment(Vector3::new(p0.x, p0.y, p0.z), d, 0.00001, length - 0.00001);
        self.rtscene
            .occluded(&mut intersection_ctx, &mut embree_ray);
        embree_ray.tfar == std::f32::NEG_INFINITY
    }
}

/// Scene representation
pub struct Scene {
    /// Main camera
    pub camera: Camera,
    pub nb_samples: usize,
    pub nb_threads: Option<usize>,
    pub output_img_path: String,
    // Geometry information
    pub meshes: Vec<geometry::Mesh>,
    pub emitter_environment: Option<EnvironmentLight>,
}

impl Scene {
    pub fn output_img(mut self, filename: &str) -> Self {
        self.output_img_path = filename.to_string();
        self
    }
    pub fn nb_threads(mut self, n: usize) -> Self {
        self.nb_threads = Some(n);
        self
    }
    pub fn nb_samples(mut self, n: usize) -> Self {
        self.nb_samples = n;
        self
    }

    pub fn emitters_sampler<'scene>(&'scene self) -> EmitterSampler<'scene> {
        // Append emission mesh to the emitter list
        let mut emitters: Vec<&dyn Emitter> = vec![];
        for e in &self.meshes {
            if !e.emission.is_zero() {
                emitters.push(e)
            }
        }
        // Construct the CDF for all the emitters
        let emitters_cdf = {
            let mut cdf_construct = Distribution1DConstruct::new(emitters.len());
            emitters
                .iter()
                .map(|e| e.flux())
                .for_each(|f| cdf_construct.add(f.channel_max()));
            cdf_construct.normalize()
        };

        EmitterSampler {
            emitters,
            emitters_cdf,
        }
    }

    pub fn enviroment_luminance(&self, d: Vector3<f32>) -> Color {
        match self.emitter_environment {
            None => Color::zero(),
            Some(ref env) => env.emitted_luminance(d),
        }
    }
}
