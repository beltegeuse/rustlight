use cgmath::InnerSpace;
use cgmath::Point2;
use integrators::*;
use paths::path::*;
use paths::vertex::*;
use samplers;
use std::cell::RefCell;
use std::rc::Rc;
use structure::*;

pub struct IntegratorLightTracing {
    pub max_depth: Option<u32>,
}

/// This structure is responsible to the graph generation
pub struct TechniqueLightTracing {
    pub max_depth: Option<u32>,
    pub samplings: Vec<Box<SamplingStrategy>>,
    pub pdf_vertex: Option<PDF>,
}

impl<'a> Technique<'a> for TechniqueLightTracing {
    fn init(
        &mut self,
        scene: &'a Scene,
        sampler: &mut Sampler,
    ) -> Vec<(Rc<RefCell<Vertex<'a>>>, Color)> {
        let (mesh, pdf, sampled_point) =
            scene.random_sample_emitter_position(sampler.next(), sampler.next(), sampler.next2d());
        let emitter_vertex = Rc::new(RefCell::new(Vertex::Emitter(EmitterVertex {
            pos: sampled_point.p,
            n: sampled_point.n,
            mesh,
            edge_in: None,
            edge_out: None,
        })));
        self.pdf_vertex = Some(pdf); // Capture the pdf for later evaluation
        vec![(emitter_vertex, Color::one())]
    }

    fn expand(&self, _vertex: &Rc<RefCell<Vertex<'a>>>) -> bool {
        return true;
    }

    fn strategies(&self, _vertex: &Rc<RefCell<Vertex<'a>>>) -> &Vec<Box<SamplingStrategy>> {
        &self.samplings
    }
}
impl TechniqueLightTracing {
    fn evaluate<'a>(
        &self,
        scene: &'a Scene,
        vertex: &Rc<VertexPtr<'a>>,
        bitmap: &mut Bitmap,
        flux: Color,
    ) {
        match *vertex.borrow() {
            Vertex::Surface(ref v) => {
                // Chech the visibility from the point to the sensor
                let pos_sensor = scene.camera.param.pos;
                let d = (pos_sensor - v.its.p).normalize();
                // Splat the contribution
                if let Some((importance, uv)) = scene.camera.sample_direct(&v.its.p) {
                    let bsdf_value = v.its.mesh.bsdf.eval(
                        &v.its.uv,
                        &v.its.wi,
                        &v.its.frame.to_local(d),
                    );
                    bitmap.accumulate(
                        Point2::new(uv.x as u32, uv.y as u32),
                        flux,
                        &"primal".to_string(),
                    ); // * importance * bsdf_value
                }

                for edge in &v.edge_out {
                    let contrib = edge.borrow().contribution();
                    if !contrib.is_zero() {
                        let edge = edge.borrow();
                        if let Some(ref vertex_next) = edge.vertices.1 {
                            self.evaluate(
                                scene,
                                vertex_next,
                                bitmap,
                                flux * edge.weight * edge.rr_weight,
                            );
                        }
                    }
                }
            }
            Vertex::Emitter(ref v) => {
                let flux = Color::one(); //v.mesh.emission / self.pdf_vertex.as_ref().unwrap().value();
                if let Some(ref edge) = v.edge_out {
                    let edge = edge.borrow();
                    if let Some(ref next_vertex) = edge.vertices.1 {
                        self.evaluate(scene, next_vertex, bitmap, flux);
                    }
                }
            }
            _ => {}
        }
    }
}

impl Integrator for IntegratorLightTracing {
    fn compute(&self, scene: &Scene) -> Bitmap {
        // Number of samples that the system will trace
        let nb_threads = rayon::current_num_threads();
        let nb_jobs = nb_threads * 4;
        let mut samplers = Vec::new();
        for _ in 0..nb_jobs {
            samplers.push(samplers::independent::IndependentSampler::default());
        }
        let nb_samples = (scene.nb_samples()
            * ((scene.camera.size().x * scene.camera.size().y) as usize)
            / nb_jobs) as usize;

        info!("Rendering...");
        let start = Instant::now();
        let progress_bar = Mutex::new(ProgressBar::new(samplers.len() as u64));
        let buffer_names = vec!["primal".to_string()];
        let img = Mutex::new(Bitmap::new(
            Point2::new(0, 0),
            *scene.camera.size(),
            &buffer_names,
        ));

        samplers.par_iter_mut().for_each(|s| {
            let mut my_img: Bitmap =
                Bitmap::new(Point2::new(0, 0), *scene.camera.size(), &buffer_names);
            (0..nb_samples).into_iter().for_each(|_| {
                // The sampling strategies
                let samplings: Vec<Box<SamplingStrategy>> =
                    vec![Box::new(DirectionalSamplingStrategy {})];
                // Do the sampling here
                let mut technique = TechniqueLightTracing {
                    max_depth: self.max_depth,
                    samplings,
                    pdf_vertex: None,
                };
                let root = generate(scene, s, &mut technique);
                // Evaluate the path generated using camera splatting operation
                technique.evaluate(scene, &root[0].0, &mut my_img, Color::one());
            });
            my_img.scale(1.0 / (nb_samples as f32));
            {
                img.lock().unwrap().accumulate_bitmap(&my_img);
                progress_bar.lock().unwrap().inc();
            }
        });

        let mut img: Bitmap = img.into_inner().unwrap();
        let elapsed = start.elapsed();
        info!(
            "Elapsed: {} ms",
            (elapsed.as_secs() * 1_000) + (elapsed.subsec_nanos() / 1_000_000) as u64
        );

        img.scale(1.0 / nb_jobs as f32);
        img
    }
}
