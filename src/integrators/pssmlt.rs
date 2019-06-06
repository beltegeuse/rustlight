use crate::integrators::*;
use crate::samplers;
use crate::structure::*;
use crate::scene::*;
use cgmath::Point2;

struct MCMCState {
    pub value: Color,
    pub tf: f32,
    pub pix: Point2<u32>,
    pub weight: f32,
}

impl MCMCState {
    pub fn new(v: Color, pix: Point2<u32>) -> MCMCState {
        MCMCState {
            value: v,
            tf: (v.r + v.g + v.b) / 3.0,
            pix,
            weight: 0.0,
        }
    }

    pub fn color(&self) -> Color {
        self.value * (self.weight / self.tf)
    }
}

pub struct IntegratorPSSMLT {
    pub large_prob: f32,
    pub integrator: Box<IntegratorMC>,
}
impl Integrator for IntegratorPSSMLT {
    fn compute(&mut self, scene: &Scene) -> BufferCollection {
        ///////////// Define the closure
        let sample = |s: &mut Sampler, emitters: &EmitterSampler| {
            let x = (s.next() * scene.camera.size().x as f32) as u32;
            let y = (s.next() * scene.camera.size().y as f32) as u32;
            let c = { self.integrator.compute_pixel((x, y), scene, s, emitters) };
            MCMCState::new(c, Point2::new(x, y))
        };

        ///////////// Compute the normalization factor
        info!("Computing normalization factor...");
        let b = self.compute_normalization(scene, 10000);
        info!("Normalisation factor: {:?}", b);

        ///////////// Compute the state initialization
        let nb_samples_total =
            scene.nb_samples * (scene.camera.size().x * scene.camera.size().y) as usize;
        let nb_samples_per_chains = 100_000;
        let nb_chains = nb_samples_total / nb_samples_per_chains;
        info!("Number of states: {:?}", nb_chains);
        // - Initialize the samplers
        let mut samplers = Vec::new();
        for _ in 0..nb_chains {
            samplers.push(samplers::mcmc::IndependentSamplerReplay::default());
        }

        ///////////// Compute the rendering (with the number of samples)
        info!("Rendering...");
        let start = Instant::now();
        let progress_bar = Mutex::new(ProgressBar::new(samplers.len() as u64));
        let buffer_names = vec!["primal".to_string()];
        let img = Mutex::new(BufferCollection::new(
            Point2::new(0, 0),
            *scene.camera.size(),
            &buffer_names,
        ));
        let pool = generate_pool(scene);
        pool.install(|| {
            samplers.par_iter_mut().for_each(|s| {
                let emitters = scene.emitters_sampler();
                // Initialize the sampler
                s.large_step = true;
                let mut current_state = sample(s as &mut Sampler, &emitters);
                while current_state.tf == 0.0 {
                    s.reject();
                    current_state = sample(s as &mut Sampler, &emitters);
                }
                s.accept();

                let mut my_img: BufferCollection =
                    BufferCollection::new(Point2::new(0, 0), *scene.camera.size(), &buffer_names);
                (0..nb_samples_per_chains).for_each(|_| {
                    // Choose randomly between large and small perturbation
                    s.large_step = s.rand() < self.large_prob;
                    let mut proposed_state = sample(s, &emitters);
                    let accept_prob = (proposed_state.tf / current_state.tf).min(1.0);
                    // Do waste reclycling
                    current_state.weight += 1.0 - accept_prob;
                    proposed_state.weight += accept_prob;
                    if accept_prob > s.rand() {
                        my_img.accumulate(
                            current_state.pix,
                            current_state.color(),
                            &buffer_names[0],
                        );
                        s.accept();
                        current_state = proposed_state;
                    } else {
                        my_img.accumulate(
                            proposed_state.pix,
                            proposed_state.color(),
                            &buffer_names[0],
                        );
                        s.reject();
                    }
                });
                // Flush the last state
                my_img.accumulate(current_state.pix, current_state.color(), &buffer_names[0]);

                my_img.scale(1.0 / (nb_samples_per_chains as f32));
                {
                    img.lock().unwrap().accumulate_bitmap(&my_img);
                    progress_bar.lock().unwrap().inc();
                }
            });
        });

        let mut img: BufferCollection = img.into_inner().unwrap();
        let elapsed = start.elapsed();
        info!("Elapsed: {:?}", elapsed,);

        // ==== Compute and scale to the normalization factor
        let img_avg = img.average_pixel(&buffer_names[0]);
        let img_avg_lum = (img_avg.r + img_avg.g + img_avg.b) / 3.0;
        img.scale(b / img_avg_lum);

        img
    }
}
impl IntegratorPSSMLT {
    fn compute_normalization(&self, scene: &Scene, nb_samples: usize) -> f32 {
        assert_ne!(nb_samples, 0);

        let mut sampler = samplers::independent::IndependentSampler::default();
        (0..nb_samples)
            .map(|_i| {
                let emitters = scene.emitters_sampler();
                let x = (sampler.next() * scene.camera.size().x as f32) as u32;
                let y = (sampler.next() * scene.camera.size().y as f32) as u32;
                let c = self.integrator.compute_pixel((x, y), scene, &mut sampler, &emitters);
                (c.r + c.g + c.b) / 3.0
            })
            .sum::<f32>()
            / (nb_samples as f32)
    }
}
