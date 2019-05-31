use crate::camera::Camera;
use crate::emitter::*;
use crate::geometry;
use crate::math::Distribution1D;
use crate::structure::*;
use cgmath::*;
use embree_rs;
use std::sync::Arc;

/// Scene representation
pub struct Scene {
    /// Main camera
    pub camera: Camera,
    pub nb_samples: usize,
    pub nb_threads: Option<usize>,
    pub output_img_path: String,
    // Geometry information
    pub meshes: Vec<Arc<geometry::Mesh>>,
    pub emitters: Vec<Arc<dyn Emitter>>,
    pub emitters_cdf: Distribution1D,
    pub embree_scene: embree_rs::Scene,
    pub emitter_environment: Option<Arc<EnvironmentLight>>,
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
        // FIXME: Check the writing for m
        let emitter_id = self
            .emitters
            .iter()
            .position(|m| &(**m) as *const _ == light_sampling.emitter as *const _)
            .unwrap();
        light_sampling.emitter.direct_pdf(&light_sampling) * self.emitters_cdf.pdf(emitter_id)
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
    ) -> (&dyn Emitter, PDF, SampledPosition) {
        let (pdf_sel, emitter) = self.random_select_emitter(v1);
        let (pdf, sampled_pos) = emitter.sample_position(v2, uv);
        (emitter, pdf * pdf_sel, sampled_pos)
    }

    pub fn enviroment_luminance(&self, d: Vector3<f32>) -> Color {
        match self.emitter_environment {
            None => Color::zero(),
            Some(ref env) => env.emitted_luminance(d),
        }
    }
}
