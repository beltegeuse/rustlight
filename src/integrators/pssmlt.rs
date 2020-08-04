use crate::integrators::*;
use crate::math::*;
use crate::samplers;
use cgmath::Point2;
use rand::rngs::SmallRng;
use rayon::prelude::*;

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

/// Function to compute the normalization factor
fn compute_normalization<F>(
    nb_samples: usize,
    routine: F,
) -> (f32, Vec<(f32, SmallRng)>, Distribution1D)
where
    F: Fn(&mut dyn Sampler) -> MCMCState,
{
    assert_ne!(nb_samples, 0);

    // TODO: Here we do not need to change the way to sample the image space
    //  As there is no burning period implemented.
    let mut sampler = crate::samplers::independent::IndependentSampler::default();
    let mut seeds = vec![];

    // Generate seeds
    for _ in 0..nb_samples {
        let current_seed = sampler.rnd.clone();
        let state = routine(&mut sampler);
        if state.tf > 0.0 {
            seeds.push((state.tf, current_seed));
        }
    }

    let mut cdf = Distribution1DConstruct::new(seeds.len());
    for s in &seeds {
        cdf.add(s.0);
    }
    let cdf = cdf.normalize();
    let b = cdf.normalization / nb_samples as f32;
    if b == 0.0 {
        panic!("Normalization is 0, impossible to continue");
    }

    (b, seeds, cdf)
}

pub struct IntegratorPSSMLT {
    pub large_prob: f32,
    pub nb_samples_norm: usize,
    pub integrator: Box<dyn IntegratorMC>,
}
impl Integrator for IntegratorPSSMLT {
    fn compute(
        &mut self,
        _: &mut dyn Sampler,
        accel: &dyn Acceleration,
        scene: &Scene,
    ) -> BufferCollection {
        ///////////// Define the closure
        let sample = |s: &mut dyn Sampler, emitters: &EmitterSampler| {
            let x = (s.next() * scene.camera.size().x as f32) as u32;
            let y = (s.next() * scene.camera.size().y as f32) as u32;
            let c = {
                self.integrator
                    .compute_pixel((x, y), accel, scene, s, emitters)
            };
            MCMCState::new(c, Point2::new(x, y))
        };

        ///////////// Compute the normalization factor
        info!("Computing normalization factor...");
        let (b, seeds, cdf) = {
            let emitters = scene.emitters_sampler();
            let sample_uni = |s: &mut dyn Sampler| -> MCMCState { sample(s, &emitters) };

            compute_normalization(self.nb_samples_norm, sample_uni)
        };
        info!("Normalisation factor: {:?}", b);
        info!("Number of *potential* seeds: {}", seeds.len());

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
            samplers.par_iter_mut().enumerate().for_each(|(id, s)| {
                let emitters = scene.emitters_sampler();

                // Initialize the sampler
                s.large_step = true;
                let previous_rnd = s.rnd.clone(); // We save the RNG (to recover it later)
                                                  // Use deterministic sampling to select a given seed
                let id_v = (id as f32 + 0.5) / nb_chains as f32;
                let seed = &seeds[cdf.sample(id_v)];
                // Replace the seed, check that the target function values matches
                s.rnd = seed.1.clone();
                let mut current_state = sample(s, &emitters);
                if current_state.tf != seed.0 {
                    error!(
                        "Unconsitency found when seeding the chain {} ({})",
                        current_state.tf, seed.0
                    );
                    return; // Stop this chain. Maybe consider completely stop the program.
                }
                s.accept();
                // We replace the RNG again. This is important as two chain
                // selective the same seed need to be independent
                s.rnd = previous_rnd;

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
