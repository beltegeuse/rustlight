use crate::emitter::*;
use crate::integrators::*;
use crate::paths::path::*;
use crate::paths::vertex::*;
use crate::samplers;
use crate::structure::*;
use cgmath::InnerSpace;
use cgmath::Point2;
use std::cell::RefCell;
use std::rc::Rc;

pub struct IntegratorLightTracing {
    pub max_depth: Option<u32>,
}

/// This structure is responsible to the graph generation
pub struct TechniqueLightTracing {
    pub max_depth: Option<u32>,
    pub samplings: Vec<Box<SamplingStrategy>>,
    pub pdf_vertex: Option<PDF>,
}

impl Technique for TechniqueLightTracing {
    fn init<'scene, 'emitter>(
        &mut self,
        path: &mut Path<'scene, 'emitter>,
        scene: &'scene Scene,
        sampler: &mut Sampler,
        emitters: &'emitter EmitterSampler,
    ) -> Vec<(VertexID, Color)> {
        let (emitter, sampled_point) = emitters.random_sample_emitter_position(
            sampler.next(),
            sampler.next(),
            sampler.next2d(),
        );
        let emitter_vertex = Vertex::Light(EmitterVertex {
            pos: sampled_point.p,
            n: sampled_point.n,
            emitter,
            edge_in: None,
            edge_out: None,
        });
        self.pdf_vertex = Some(sampled_point.pdf); // Capture the pdf for later evaluation
        vec![(path.register_vertex(emitter_vertex), Color::one())]
    }

    fn expand(&self, vertex: &Vertex, depth: u32) -> bool {
        self.max_depth.map_or(true, |max| depth < max)
    }

    fn strategies(&self, vertex: &Vertex) -> &Vec<Box<SamplingStrategy>> {
        &self.samplings
    }
}
impl TechniqueLightTracing {
    fn evaluate<'scene>(
        &self,
        path: &Path<'scene, '_>,
        scene: &'scene Scene,
        vertex_id: VertexID,
        bitmap: &mut BufferCollection,
        flux: Color,
    ) {
        match path.vertex(vertex_id) {
            Vertex::Surface(ref v) => {
                // Chech the visibility from the point to the sensor
                let pos_sensor = scene.camera.position();
                let d = (pos_sensor - v.its.p).normalize();
                if !v.its.mesh.bsdf.is_smooth() && scene.visible(&v.its.p, &pos_sensor) {
                    // Splat the contribution
                    if let Some((importance, uv)) = scene.camera.sample_direct(&v.its.p) {
                        // Compute BSDF for the splatting
                        let wo_local = v.its.frame.to_local(d);
                        let wi_global = v.its.frame.to_world(v.its.wi);
                        let bsdf_value = v.its.mesh.bsdf.eval(
                            &v.its.uv,
                            &v.its.wi,
                            &wo_local,
                            Domain::SolidAngle,
                        );
                        let correction = (v.its.wi.z * d.dot(v.its.n_g))
                            / (wo_local.z * wi_global.dot(v.its.n_g));
                        // Accumulate the results
                        bitmap.accumulate_safe(
                            Point2::new(uv.x as i32, uv.y as i32),
                            flux * importance * bsdf_value * correction,
                            &"primal".to_owned(),
                        );
                    }
                }

                for edge_id in &v.edge_out {
                    let edge = path.edge(*edge_id);
                    if let Some(vertex_next) = edge.vertices.1 {
                        self.evaluate(
                            path,
                            scene,
                            vertex_next,
                            bitmap,
                            flux * edge.weight * edge.rr_weight,
                        );
                    }
                }
            }
            Vertex::Light(ref v) => {
                let flux = v.emitter.flux() / self.pdf_vertex.as_ref().unwrap().value();
                let pos_sensor = scene.camera.position();
                let d = (pos_sensor - v.pos).normalize();
                if scene.visible(&v.pos, &pos_sensor) {
                    if let Some((importance, uv)) = scene.camera.sample_direct(&v.pos) {
                        bitmap.accumulate_safe(
                            Point2::new(uv.x as i32, uv.y as i32),
                            flux * importance * d.dot(v.n),
                            &"primal".to_owned(),
                        );
                    }
                }
                if let Some(edge_id) = v.edge_out {
                    let edge = path.edge(edge_id);
                    if let Some(next_vertex) = edge.vertices.1 {
                        self.evaluate(path, scene, next_vertex, bitmap, edge.weight * flux);
                    }
                }
            }
            _ => {}
        }
    }
}

impl Integrator for IntegratorLightTracing {
    fn compute(&mut self, scene: &Scene) -> BufferCollection {
        // Number of samples that the system will trace
        let nb_threads = rayon::current_num_threads();
        let nb_jobs = nb_threads * 4;
        let mut samplers = Vec::new();
        for _ in 0..nb_jobs {
            samplers.push(samplers::independent::IndependentSampler::default());
        }
        let nb_samples = (scene.nb_samples
            * ((scene.camera.size().x * scene.camera.size().y) as usize))
            / nb_jobs as usize;

        let progress_bar = Mutex::new(ProgressBar::new(samplers.len() as u64));
        let buffer_names = vec![String::from("primal")];
        let img = Mutex::new(BufferCollection::new(
            Point2::new(0, 0),
            *scene.camera.size(),
            &buffer_names,
        ));

        let pool = generate_pool(scene);
        pool.install(|| {
            samplers.par_iter_mut().for_each(|s| {
                let mut my_img: BufferCollection =
                    BufferCollection::new(Point2::new(0, 0), *scene.camera.size(), &buffer_names);
                let emitters = scene.emitters_sampler();
                (0..nb_samples).for_each(|_| {
                    // The sampling strategies
                    let samplings: Vec<Box<SamplingStrategy>> =
                        vec![Box::new(DirectionalSamplingStrategy {})];
                    // Do the sampling here
                    let mut technique = TechniqueLightTracing {
                        max_depth: self.max_depth,
                        samplings,
                        pdf_vertex: None,
                    };
                    let mut path = Path::default();
                    let root = generate(&mut path, scene, &emitters, s, &mut technique);
                    // Evaluate the path generated using camera splatting operation
                    technique.evaluate(&path, scene, root[0].0, &mut my_img, Color::one());
                });
                my_img.scale(1.0 / (nb_samples as f32));
                {
                    img.lock().unwrap().accumulate_bitmap(&my_img);
                    progress_bar.lock().unwrap().inc();
                }
            });
        });

        let mut img: BufferCollection = img.into_inner().unwrap();
        img.scale(1.0 / nb_jobs as f32);
        img.scale((scene.camera.img.x * scene.camera.img.y) as f32);
        img
    }
}
