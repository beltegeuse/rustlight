use crate::camera::Camera;
use crate::emitter::*;
use crate::geometry;
use crate::math::Distribution1DConstruct;
use crate::structure::*;
use cgmath::*;

pub struct BasicIntersection {
    pub n_g: Vector3<f32>,  //< Geometric normal
    pub uv: Vector2<f32>,   //< barycentric coordinate
    pub t: f32,             //< Distance intersection
    pub id_geo: usize,      //< ID geometry
    pub id_prim: usize,     //< ID primitive 
    pub id_inst: usize,     //< ID instance
}
pub trait Acceleration: Sync {
    fn trace(&self, ray: &Ray) -> Option<BasicIntersection>; 
    fn visible(&self, p0: &Point3<f32>, p1: &Point3<f32>) -> bool;
}

pub struct EmbreeAcceleration<'scene> {
    pub rtscene: embree_rs::CommittedScene<'scene>,
}
impl<'scene> EmbreeAcceleration<'scene> {
    pub fn new(scene: &'scene embree_rs::Scene) -> Self {
        let rtscene = scene.commit();
        EmbreeAcceleration {
            rtscene
        }
    }
}

impl<'scene> Acceleration for EmbreeAcceleration<'scene> {
    fn trace(&self, ray: &Ray) -> Option<BasicIntersection> {
        let mut intersection_ctx = embree_rs::IntersectContext::coherent();
        let ray = embree_rs::Ray::segment(Vector3::new(ray.o.x,ray.o.y,ray.o.z), ray.d, ray.tnear, ray.tfar);
        let mut ray_hit = embree_rs::RayHit::new(ray);
        self.rtscene.intersect(&mut intersection_ctx, &mut ray_hit);
        if ray_hit.hit.hit() {
           Some(BasicIntersection {
               n_g: Vector3::new(ray_hit.hit.Ng_x,ray_hit.hit.Ng_y,ray_hit.hit.Ng_z),
               uv: Vector2::new(ray_hit.hit.u, ray_hit.hit.v),
               t: ray_hit.ray.tfar,
               id_geo: ray_hit.hit.geomID as usize,
               id_prim: ray_hit.hit.primID as usize,
               id_inst: ray_hit.hit.instID[0] as usize,
           })
        } else {
            None
        }
    }
    fn visible(&self, p0: &Point3<f32>, p1: &Point3<f32>) -> bool {
        unimplemented!()
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
    // Acceleration for intersection
    pub acceleration: Box<Acceleration>,
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

    /// Intersect and compute intersection information
    pub fn trace(&self, ray: &Ray) -> Option<Intersection> {
        match self.acceleration.trace(ray) {
            None => None,
            Some(basic_info) => {
                unimplemented!()
                // Retrive the mesh
                // let n_s = if basic_info.n_s.is_none() {
                //     embree_its.n_g
                // } else {
                //     embree_its.n_s.unwrap()
                // };
                
                // // TODO: Hack for now for make automatic twosided.
                // let (n_s, n_g) = if mesh.bsdf.is_twosided() && mesh.emission.is_zero() && d.dot(n_s) <= 0.0
                // {
                //     (
                //         Vector3::new(-n_s.x, -n_s.y, -n_s.z),
                //         Vector3::new(-embree_its.n_g.x, -embree_its.n_g.y, -embree_its.n_g.z),
                //     )
                // } else {
                //     (n_s, embree_its.n_g)
                // };

                // let frame = Frame::new(n_s);
                // let wi = frame.to_local(d);
                // Some(Intersection {
                //     dist: embree_its.t,
                //     n_g,
                //     n_s,
                //     p: embree_its.p,
                //     uv: embree_its.uv,
                //     mesh,
                //     frame,
                //     wi,
                // })
            }
        }
    }
    pub fn visible(&self, p0: &Point3<f32>, p1: &Point3<f32>) -> bool {
        self.visible(p0, p1)
    }

    pub fn enviroment_luminance(&self, d: Vector3<f32>) -> Color {
        match self.emitter_environment {
            None => Color::zero(),
            Some(ref env) => env.emitted_luminance(d),
        }
    }
}
