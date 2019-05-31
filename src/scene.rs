use crate::camera::Camera;
use crate::emitter::*;
use crate::geometry;
use crate::math::Distribution1D;
use crate::structure::*;
use crate::math::Distribution1DConstruct;
use cgmath::*;
use embree_rs;
use std::sync::Arc;

/// Scene representation
pub struct Scene<'scene> {
    /// Main camera
    pub camera: Camera,
    pub nb_samples: usize,
    pub nb_threads: Option<usize>,
    pub output_img_path: String,
    // Geometry information
    pub meshes: Vec<Arc<geometry::Mesh>>,
    pub emitters: Vec<&'scene dyn Emitter>,
    pub emitters_cdf: Option<Distribution1D>,
    pub embree_scene: embree_rs::Scene,
    pub emitter_environment: Option<Arc<EnvironmentLight>>,
}

impl<'scene> Scene<'scene> {
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

    pub fn configure(&mut self) {
        // Append emission mesh to the emitter list
        self.emitters.append(self.meshes
            .iter()
            .filter(|m| !m.emission.is_zero())
            .map(|m| {
                let m: &dyn Emitter = &(**m);
                m 
            })
            .collect::<Vec<_>>());
        
        // Construct the CDF for all the emitters
        let emitters_cdf = {
            let mut cdf_construct = Distribution1DConstruct::new(self.emitters.len());
            self.emitters
                .iter()
                .map(|e| e.flux())
                .for_each(|f| cdf_construct.add(f.channel_max()));
            cdf_construct.normalize()
        };
        info!(
            "CDF lights: {:?} norm: {:?}",
            emitters_cdf.cdf, emitters_cdf.normalization
        );
        // Setup the emitter CDF for the current scene
        self.emitters_cdf = Some(emitters_cdf);
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
        !self
            .embree_scene
            .occluded(embree_rs::Ray::new(*p0, d).near(0.00001).far(0.9999))
    }

    pub fn direct_pdf(&self, light_sampling: &LightSamplingPDF) -> PDF {
        let emitter_cdf = self.emitters_cdf.as_ref().unwrap();

        // FIXME: Check the writing for m
        let emitter_id = match self
            .emitters
            .iter()
            .position(|m| m as *const _ == light_sampling.emitter as *const _)
        {
            Some(v) => v,
            None => {
                panic!(
                    "Impossible to found the emitter ID inside the CDF: {:p}",
                    light_sampling.emitter
                );
            }
        };
        light_sampling.emitter.direct_pdf(&light_sampling) * emitter_cdf.pdf(emitter_id)
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
        let emitter_cdf = self.emitters_cdf.as_ref().unwrap();
        let id_light = emitter_cdf.sample(v);
        (emitter_cdf.pdf(id_light), &(*self.emitters[id_light]))
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

    pub fn enviroment_luminance(&self, d: Vector3<f32>) -> Color {
        match self.emitter_environment {
            None => Color::zero(),
            Some(ref env) => env.emitted_luminance(d),
        }
    }
}
