use crate::camera::Camera;
use crate::emitter::*;
use crate::geometry;
use crate::math::Distribution1DConstruct;
use crate::structure::*;
use crate::volume;
use cgmath::*;
use std::sync::Arc;

pub enum EmittersState {
    Unbuild(Vec<Box<dyn Emitter>>),
    Build(EmitterSampler),
}

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
    // Note that we need an option to call take()
    pub emitters: Option<EmittersState>,
    pub bsphere: Option<BoundingSphere>,
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
        match &self.emitters {
            Some(EmittersState::Build(s)) => s,
            _ => panic!("The emitters are not built?"),
        }
    }

    pub fn build_emitters(&mut self) {
        // Compute the bounding box
        let mut aabb = AABB::default();
        for m in &self.meshes {
            aabb = aabb.union_aabb(&m.compute_aabb());
        }
        aabb = aabb.union_vec(&self.camera.position().to_vec());
        self.bsphere = Some(aabb.to_sphere());

        // Append emission mesh to the emitter list
        let mut emitters: Vec<Arc<dyn Emitter>> = vec![];
        for e in &self.meshes {
            if !e.emission.is_zero() {
                emitters.push(e.clone())
            }
        }
        // Add env map
        if self.emitter_environment.is_some() {
            // Need to take and remove the Arc to be able to modify
            // It is fine as we will have only one version of the scene
            let envmap = self.emitter_environment.take().unwrap();
            let mut envmap = Arc::try_unwrap(envmap).unwrap();
            envmap.preprocess(self);

            // Now it is initialized, we need to put it back inside emitter_env
            self.emitter_environment = Some(Arc::new(envmap));
            // Add it to the list of potential emitters to samples
            emitters.push(self.emitter_environment.as_ref().unwrap().clone());
        }

        // Add other emitters
        let other_emitters = self.emitters.take();
        match other_emitters {
            Some(EmittersState::Build(_)) => panic!("Emitters are build twice?"),
            None => {}
            Some(EmittersState::Unbuild(other_emitters)) => {
                for mut e in other_emitters {
                    e.preprocess(self);
                    emitters.push(Arc::from(e));
                }
            }
        };

        // Stop early if necessary
        if emitters.is_empty() {
            warn!("No emitter detected, if NEE is called, rustlight will certainly crashing");
            return;
        }

        // Construct the CDF for all the emitters
        let emitters_cdf = {
            let mut cdf_construct = Distribution1DConstruct::new(emitters.len());
            emitters
                .iter()
                .map(|e| e.flux())
                .for_each(|f| cdf_construct.add(f.channel_max()));
            dbg!(&cdf_construct);
            cdf_construct.normalize()
        };

        self.emitters = Some(EmittersState::Build(EmitterSampler {
            emitters,
            emitters_cdf,
        }));
    }

    pub fn enviroment_luminance(&self, d: Vector3<f32>) -> Color {
        match self.emitter_environment {
            None => Color::zero(),
            Some(ref env) => env.eval(d),
        }
    }
}
