use crate::camera::Camera;
use crate::emitter::*;
use crate::geometry;
use crate::math::Distribution1D;
use crate::math::Distribution1DConstruct;
use crate::structure::*;
use cgmath::*;

/// Scene representation
pub struct Scene {
    /// Main camera
    pub camera: Camera,
    pub nb_samples: usize,
    pub nb_threads: Option<usize>,
    pub output_img_path: String,
    // Geometry information
    pub meshes: Vec<geometry::Mesh>,
    pub embree_scene: embree_rs::Scene,
    pub emitter_environment: Option<EnvironmentLight>,
}

pub struct EmitterSampler<'scene> {
    pub emitters: Vec<&'scene dyn Emitter>,
    pub emitters_cdf: Distribution1D,
}

impl<'scene> EmitterSampler<'scene> {
    pub fn pdf(&self, emitter: &dyn Emitter) -> f32 {
        let emitter_addr: [usize; 2] = unsafe { std::mem::transmute(emitter) };
        for (i, e) in self.emitters.iter().enumerate() {
            let other_addr: [usize; 2] = unsafe { std::mem::transmute(*e) };
            if emitter_addr[0] == other_addr[0] {
                //if std::ptr::eq(emitter, *e) {
                // I need the index to retrive an info
                // This info cannot be stored inside the Emitter
                return self.emitters_cdf.pdf(i);
            }
        }

        // For debug
        println!("Size: {}", self.emitters.len());
        for e in &self.emitters {
            println!(" - {:p} != {:p}", (*e), emitter);
        }

        panic!("Impossible to found the emitter: {:p}", emitter);
    }
    pub fn sample_light(
        &self,
        p: &Point3<f32>,
        r_sel: f32,
        r: f32,
        uv: Point2<f32>,
    ) -> LightSampling {
        // Select the point on the light
        let (pdf_sel, emitter) = self.random_select_emitter(r_sel);
        let mut res = emitter.sample_direct(p, r, uv);
        res.pdf = res.pdf * pdf_sel;
        res
    }
    pub fn random_select_emitter(&self, v: f32) -> (f32, &dyn Emitter) {
        let id_light = self.emitters_cdf.sample(v);
        (self.emitters_cdf.pdf(id_light), &(*self.emitters[id_light]))
    }

    pub fn random_sample_emitter_position(
        &self,
        v1: f32,
        v2: f32,
        uv: Point2<f32>,
    ) -> (&dyn Emitter, SampledPosition) {
        let (pdf_sel, emitter) = self.random_select_emitter(v1);
        let mut sampled_pos = emitter.sample_position(v2, uv);
        sampled_pos.pdf = sampled_pos.pdf * pdf_sel;
        (emitter, sampled_pos)
    }
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
        match self.embree_scene.intersect(ray.to_embree()) {
            None => None,
            Some(its) => {
                let geom_id = its.geom_id as usize;
                Some(Intersection::new(&its, -ray.d, &self.meshes[geom_id]))
            }
        }
    }
    pub fn visible(&self, p0: &Point3<f32>, p1: &Point3<f32>) -> bool {
        let d = p1 - p0;
        !self.embree_scene
            .occluded(embree_rs::Ray::new(*p0, d).near(0.00001).far(0.9999))
    }

    pub fn enviroment_luminance(&self, d: Vector3<f32>) -> Color {
        match self.emitter_environment {
            None => Color::zero(),
            Some(ref env) => env.emitted_luminance(d),
        }
    }
}
