use crate::integrators::mcmc::*;
use crate::integrators::*;
use crate::samplers::mcmc::IndependentSamplerReplay;
use cgmath::Point2;
use rand::SeedableRng;
use rand::*;

pub struct IntegratorERPT {
    /// Control how much sample budget is dedicated to exploration
    /// This parameter control the number of chains spawn implicitely
    /// as the number of MCMC samples will be (nb_samples - nb_mc)
    pub nb_mc: usize,
    /// Number of samples produced by a chain
    /// Note that this parameter control implicitely the number of chains
    /// spawned per pixel (as well nb_mc)
    pub chain_samples: usize,
    /// Do we want to stratify the chains over the image plane
    /// This parameter should be true if you want to replicate ERPT results
    /// However, this parameter is interesting you can see the benefice of chain
    /// stratification over the image-plane
    pub stratified: bool,
    pub integrator: Box<dyn IntegratorMC>,
}

const ERPT_DEBUG: bool = false;

impl Integrator for IntegratorERPT {
    fn compute(
        &mut self,
        sampler: &mut dyn Sampler,
        accel: &dyn Acceleration,
        scene: &Scene,
    ) -> BufferCollection {
        // Make sure that we have work to do with MCMC
        assert!(self.nb_mc < scene.nb_samples);

        ///////////// Define the closure
        let sample = |s: &mut dyn Sampler| {
            let x = (s.next() * scene.camera.size().x as f32) as u32;
            let y = (s.next() * scene.camera.size().y as f32) as u32;

            // State
            let mut state = MCMCState::empty();
            let c = { self.integrator.compute_pixel((x, y), accel, scene, s) };
            state.append(c, Point2::new(x, y));
            state
        };

        // We need to remove nb_mc samples, as they will be used to do the local exploration phase...
        let spp_mcmc = scene.nb_samples - self.nb_mc;
        let nb_samples_total = spp_mcmc * (scene.camera.size().x * scene.camera.size().y) as usize;
        // --- Total number of samples
        let nb_samples_per_chains = self.chain_samples;
        // Compute the number of chains
        let nb_chains = nb_samples_total / nb_samples_per_chains;
        info!(
            "Num chains: {} | SPP per chains: {}",
            nb_chains, nb_samples_per_chains
        );
        // This is the number of chains per pixels
        let nb_chains_per_pixel =
            nb_chains as f32 / (scene.camera.size().x * scene.camera.size().y) as f32;

        // Compute the normalization factor
        // This factor is used to decide how many chains we need to spawn
        let b = average_lum(1_000_000, sample);

        // Create the full image
        let buffer_names = if ERPT_DEBUG {
            vec![
                "primal".to_string(),
                "nb_chains".to_string(),
                "mc".to_string(),
            ]
        } else {
            vec!["primal".to_string()]
        };
        let img = Mutex::new(BufferCollection::new(
            Point2::new(0, 0),
            *scene.camera.size(),
            &buffer_names,
        ));

        // Compare to classical MCMC, the sampler and the thread process is different
        // in ERPT. Indeed, we create samplers similar to IntegratorMC but as we will
        // spawn chains for all the image size, we will need the full image.
        let image_blocks = generate_img_blocks(scene, sampler, &buffer_names);
        // and replace the sampler generator with duplicate of the global sampler
        let mut image_blocks = image_blocks
            .into_iter()
            .map(|(img, _)| {
                // We will replace the sampler type to MCMC sampler
                // to be able to retain the random numbers
                let sampler = crate::samplers::mcmc::IndependentSamplerReplay::default();
                (img, sampler)
            })
            .collect::<Vec<_>>();

        ///////////// Compute the rendering (with the number of samples)
        info!("Rendering...");
        let start = Instant::now();
        let progress_bar = Mutex::new(ProgressBar::new(image_blocks.len() as u64));
        let pool = generate_pool(scene);

        // Count the number of chains spawn
        // this statistic is usefull to know the quality of the normalization factor
        let nb_chain_spawned_total = Mutex::new(0);

        pool.install(|| {
            image_blocks
                .par_iter_mut()
                .for_each(|(im_block, current_sampler)| {
                    // Create the full image
                    let mut my_img: BufferCollection = BufferCollection::new(
                        Point2::new(0, 0),
                        *scene.camera.size(),
                        &buffer_names,
                    );

                    // Function to do ERPT equal deposit rule
                    let mcmc_step =
                        |w0: f32,
                         img: &mut BufferCollection,
                         mut current_state: MCMCState,
                         s: &mut IndependentSamplerReplay| {
                            s.large_step = false; // No large steps
                            (0..nb_samples_per_chains).for_each(|_| {
                                let mut proposed_state = sample(s);
                                let accept_prob = (proposed_state.tf / current_state.tf).min(1.0);
                                // Do waste reclycling
                                current_state.weight += 1.0 - accept_prob;
                                proposed_state.weight += accept_prob;
                                let accepted = accept_prob > s.rand();
                                if accepted {
                                    current_state.accumulate_weight(img, &buffer_names[0], w0);
                                    s.accept();
                                    current_state = proposed_state;
                                } else {
                                    proposed_state.accumulate_weight(img, &buffer_names[0], w0);
                                    s.reject();
                                }
                            });
                            // Flush the last state
                            current_state.accumulate_weight(img, &buffer_names[0], w0);
                        };

                    // Do the MC steps, and if found a contributive path
                    // decide how many chains to spawn
                    let mut nb_chain_spawned = 0;
                    for iy in 0..im_block.size.y {
                        for ix in 0..im_block.size.x {
                            for _ in 0..self.nb_mc {
                                current_sampler.large_step = true;
                                let current = if self.stratified {
                                    // If stratification, we need to force the pixel sampled with MCMC
                                    // Consume the first two rng (but we will not use them)
                                    let _x = (current_sampler.next() * scene.camera.size().x as f32)
                                        as u32;
                                    let _y = (current_sampler.next() * scene.camera.size().y as f32)
                                        as u32;
                                    let mut state = MCMCState::empty();
                                    let c = {
                                        self.integrator.compute_pixel(
                                            (ix + im_block.pos.x, iy + im_block.pos.y),
                                            accel,
                                            scene,
                                            current_sampler,
                                        )
                                    };
                                    state.append(
                                        c,
                                        Point2::new(ix + im_block.pos.x, iy + im_block.pos.y),
                                    );
                                    state
                                } else {
                                    // If no stratification, do same as PSSMLT
                                    sample(current_sampler)
                                };
                                // current.tf / b (gives the scaled between the average tf and the current tf)
                                // nb_chains_per_pixel / self.nb_mc (gives the number of chains we want to starts per pixels)
                                let mean_chains = (current.tf / b)
                                    * (nb_chains_per_pixel as f32 / self.nb_mc as f32);
                                let nb_current_chains =
                                    (mean_chains + current_sampler.rand()) as u32; // Floor
                                current_sampler.accept();

                                if ERPT_DEBUG {
                                    // Track the number of mutation (in avg)
                                    my_img.accumulate(
                                        current.values[0].1,
                                        Color::value(mean_chains),
                                        &buffer_names[1],
                                    );
                                    // Track the MC estimate
                                    my_img.accumulate(
                                        current.values[0].1,
                                        current.values[0].0 / self.nb_mc as f32,
                                        &buffer_names[2],
                                    );
                                }

                                if nb_current_chains > 0 {
                                    // Equal deposit rule (b)
                                    // Here we do not have current.tf (w0) as the number of chains
                                    // will depends on this value
                                    let w0 =
                                        b / (nb_chains_per_pixel * nb_samples_per_chains as f32);
                                    nb_chain_spawned += nb_current_chains;
                                    if self.stratified {
                                        assert!(current.values.len() == 1);
                                        // Only if we have stratified, we need to remap the random number to be able
                                        // to move inside the image-plane
                                        // This is similar to "Reversible jump" but only for the sensor pixel sampling

                                        // Important note: This approach works as we know that the two first random number
                                        // are used for sampling the image plane. If it is not the case, everything will break down
                                        current_sampler.values[0].value =
                                            (current_sampler.values[0].value
                                                + current.values[0].1.x as f32)
                                                / scene.camera.size().x as f32;
                                        current_sampler.values[1].value =
                                            (current_sampler.values[1].value
                                                + current.values[0].1.y as f32)
                                                / scene.camera.size().y as f32;
                                    }
                                    for _ in 0..nb_current_chains {
                                        // We will copy the internal representation of the sampler
                                        // note that it can be a bit slow...
                                        let mut new_sampler =
                                            crate::samplers::mcmc::IndependentSamplerReplay {
                                                // We create a new sampler (to make all the chain different)
                                                rnd: rand::rngs::SmallRng::from_seed(
                                                    current_sampler.rnd.gen(),
                                                ),
                                                // Then the rest is copy from the orignal path
                                                values: current_sampler.values.clone(),
                                                backup: current_sampler.backup.clone(),
                                                mutator: current_sampler.mutator.clone_box(),
                                                time: current_sampler.time,
                                                time_large: current_sampler.time_large,
                                                indice: current_sampler.indice,
                                                large_step: false,
                                            };
                                        mcmc_step(
                                            w0,
                                            &mut my_img,
                                            current.clone(),
                                            &mut new_sampler,
                                        );
                                    }
                                }
                            }
                        }
                    }

                    // Splat the entire contribution
                    {
                        img.lock().unwrap().accumulate_bitmap(&my_img);
                        progress_bar.lock().unwrap().inc();
                        *nb_chain_spawned_total.lock().unwrap() += nb_chain_spawned;
                    }
                });
        });

        let nb_chain_spawned_total = nb_chain_spawned_total.into_inner().unwrap();
        let img: BufferCollection = img.into_inner().unwrap();
        let elapsed = start.elapsed();
        info!("Elapsed: {:?}", elapsed,);
        info!(
            "Nb Spawned chains: {} (expected: {})",
            nb_chain_spawned_total, nb_chains
        );
        if ERPT_DEBUG {
            img.dump_all("erpt.exr");
        }
        // No scaling in ERPT
        img
    }
}
