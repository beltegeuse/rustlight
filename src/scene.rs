use crate::camera::Camera;
use crate::emitter::*;
use crate::geometry;
use crate::math::Distribution1DConstruct;
use crate::structure::*;
use crate::volume;
use cgmath::*;
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
    pub emitter_environment: Option<Arc<EnvironmentLight>>,
    pub volume: Option<volume::HomogenousVolume>,
    // Internal building
    pub emitters: Option<EmitterSampler>,
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

    pub fn emitters(&self) -> &EmitterSampler {
        self.emitters.as_ref().unwrap()
    }

    pub fn build_emitters(&mut self) {
        // Append emission mesh to the emitter list
        let mut emitters: Vec<Arc<dyn Emitter>> = vec![];
        for e in &self.meshes {
            if !e.emission.is_zero() {
                emitters.push(e.clone())
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

        self.emitters = Some(EmitterSampler {
            emitters,
            emitters_cdf,
        });
    }

    pub fn enviroment_luminance(&self, d: Vector3<f32>) -> Color {
        match self.emitter_environment {
            None => Color::zero(),
            Some(ref env) => env.emitted_luminance(d),
        }
    }
}
