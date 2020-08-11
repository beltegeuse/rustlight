use crate::integrators::mcmc::*;
use crate::integrators::*;
use crate::samplers::mcmc::*;
use rayon::prelude::*;

pub const BORROW: bool = true;

// Mutator what uses random on the first dimension (AA pixel)
pub struct MutatorSMCMC {
    pub kel: MutatorKelemen,
}
impl Default for MutatorSMCMC {
    fn default() -> Self {
        Self {
            kel: MutatorKelemen::default(),
        }
    }
}
impl Mutator for MutatorSMCMC {
    fn mutate(&self, v: f32, r: f32, i: usize) -> f32 {
        match i {
            0 | 1 => r,
            _ => self.kel.mutate(v, r, i),
        }
    }
    fn clone_box(&self) -> Box<dyn Mutator> {
        Box::new(MutatorSMCMC {
            kel: MutatorKelemen {
                s1: self.kel.s1,
                s2: self.kel.s2,
                log_ratio: self.kel.log_ratio,
            },
        })
    }
}

#[derive(Debug)]
pub struct PixelValue {
    /// Position on the image plane
    pub p: Point2<u32>,
    /// Pixel's estimate by MCMC
    pub value: Color,
    /// Pixel's estimate by MC
    pub value_mc: Color,
}

impl PixelValue {
    pub fn new(p: Point2<u32>) -> Self {
        Self {
            p,
            value: Color::zero(),
            value_mc: Color::zero(),
        }
    }
}

pub struct Tile {
    pub pixels: Vec<PixelValue>,
    pub nb_samples: usize,
    // Normalization
    pub b: f32,
    pub nb_uniform: usize,
    // Scaling factor (reconstruction)
    pub scale: Color,
    // MCMC state
    pub state: Option<MCMCState>,
    // Sampler
    pub sampler: Box<crate::samplers::mcmc::IndependentSamplerReplay>,
}

impl Tile {
    fn pixels(p: Point2<u32>, img_size: &Vector2<u32>) -> Vec<PixelValue> {
        // FIXME: This function can be largely optimized
        //  by passing a mut vector or using &mut self

        // Create cross shape from the position p
        // we remove pixels that are outside the image-plane
        let mut pixels = vec![PixelValue::new(p)];
        if p.x > 0 {
            pixels.push(PixelValue::new(Point2::new(p.x - 1, p.y)));
        }
        if p.y > 0 {
            pixels.push(PixelValue::new(Point2::new(p.x, p.y - 1)));
        }
        if p.x != img_size.x - 1 {
            pixels.push(PixelValue::new(Point2::new(p.x + 1, p.y)));
        }
        if p.y != img_size.y - 1 {
            pixels.push(PixelValue::new(Point2::new(p.x, p.y + 1)));
        };
        pixels
    }

    pub fn new(p: Point2<u32>, img_size: &Vector2<u32>) -> Self {
        let mut pixels = Tile::pixels(p, img_size);
        pixels.shrink_to_fit();

        // Change the mutator to the one used in SMCMC
        let mut sampler = Box::new(crate::samplers::mcmc::IndependentSamplerReplay::default());
        sampler.mutator = Box::new(MutatorSMCMC::default());

        // Construct the object with default parameters
        Self {
            pixels,
            nb_samples: 0,
            b: 0.0,
            nb_uniform: 0,
            scale: Color::one(),
            state: None,
            sampler,
        }
    }
    /// Get the tile position (equivalent to the cross center)
    pub fn center_pos(&self) -> Point2<u32> {
        // By construction, the center pixel is always the first one
        self.pixels[0].p
    }
    pub fn reallocate(&mut self, p_new: Point2<u32>, img_size: &Vector2<u32>) {
        self.pixels.clear();
        self.pixels = Tile::pixels(p_new, img_size);
    }

    /// Method that given the tile generate a state
    pub fn generate_state<F>(&mut self, int: F) -> MCMCState
    where
        F: Fn((u32, u32), &mut dyn Sampler) -> Color,
    {
        assert_eq!(self.sampler.indice, 0);
        let mut state = MCMCState::empty();
        for (i, p) in self.pixels.iter().enumerate() {
            if i != 0 {
                self.sampler.reset_index();
            }
            let c = int((p.p.x, p.p.y), self.sampler.as_mut());
            // Uses channel max
            state.append_with_tf(c, c.channel_max(), p.p);
        }
        state
    }

    pub fn splat_state_uni(&mut self, s: &MCMCState) {
        if !self.sampler.large_step {
            panic!("Try to splat a state for normalization factor estimation generated from MCMC?");
        }
        self.b += s.tf;
        self.nb_uniform += 1;
        let w = s.weight / s.tf;
        for (i, p) in self.pixels.iter_mut().enumerate() {
            p.value_mc += s.values[i].0 * w;
        }
    }

    // This function is a duplicate of splat_state
    // to fix borrow checker issues (mut ref and ref)
    pub fn splat_state_current(&mut self) {
        let s = self.state.as_ref().unwrap();
        let w = s.weight / s.tf;
        for (i, p) in self.pixels.iter_mut().enumerate() {
            p.value += s.values[i].0 * w;
        }
        self.nb_samples += 1;
    }
    pub fn splat_state(&mut self, s: &MCMCState) {
        let w = s.weight / s.tf;
        for (i, p) in self.pixels.iter_mut().enumerate() {
            p.value += s.values[i].0 * w;
        }
        self.nb_samples += 1;
    }
}

fn chain_non_init<F>(t: &mut Tile, technique: F)
where
    F: Fn((u32, u32), &mut dyn Sampler) -> Color,
{
    t.sampler.large_step = true;
    let state = t.generate_state(technique);
    // Always accept (as it does not matter)
    t.sampler.accept();
    // Estimate normalization factor
    t.splat_state_uni(&state);
    // If we found one valid path, use it!
    if state.tf != 0.0 {
        t.state = Some(state);
    }
}

fn independent_mcmc<F>(t: &mut Tile, large_prob: f32, technique: F)
where
    F: Fn((u32, u32), &mut dyn Sampler) -> Color,
{
    t.sampler.large_step = large_prob >= t.sampler.rand();
    let mut proposed_state = t.generate_state(technique);
    if t.sampler.large_step {
        t.splat_state_uni(&proposed_state);
    }
    let accept_prob = (proposed_state.tf / t.state.as_ref().unwrap().tf).min(1.0);

    // Do waste reclycling
    t.state.as_mut().unwrap().weight += 1.0 - accept_prob;
    proposed_state.weight += accept_prob;
    let accepted = accept_prob > t.sampler.rand();
    if accepted {
        t.splat_state_current();
        t.sampler.accept();
        t.state = Some(proposed_state);
    } else {
        t.splat_state(&proposed_state);
        t.sampler.reject();
    }
}

fn independent_mcmc_safe<F>(t: &mut Tile, large_prob: f32, technique: F)
where
    F: Fn((u32, u32), &mut dyn Sampler) -> Color,
{
    if t.state.is_none() {
        chain_non_init(t, technique);
    } else {
        independent_mcmc(t, large_prob, technique);
    }
}

fn replica_exchange<F>(t0: &mut Tile, t1: &mut Tile, technique: F)
where
    F: Fn((u32, u32), &mut dyn Sampler) -> Color,
{
    // Swap the samplers
    std::mem::swap(&mut t0.sampler, &mut t1.sampler);

    // Evaluate the swapped states
    t0.sampler.large_step = false;
    t1.sampler.large_step = false;
    let mut proposed_state0 = t0.generate_state(&technique);
    let mut proposed_state1 = t1.generate_state(&technique);

    // Compute acceptance
    let accept_prob = ((proposed_state0.tf * proposed_state1.tf)
        / (t0.state.as_ref().unwrap().tf * t1.state.as_ref().unwrap().tf))
        .min(1.0);
    let accepted = accept_prob > t1.sampler.rand();

    // Do waste recycling
    proposed_state0.weight += accept_prob;
    proposed_state1.weight += accept_prob;
    t0.state.as_mut().unwrap().weight += 1.0 - accept_prob;
    t1.state.as_mut().unwrap().weight += 1.0 - accept_prob;

    if accepted {
        t0.splat_state_current();
        t1.splat_state_current();

        t0.sampler.accept();
        t1.sampler.accept();

        t0.state = Some(proposed_state0);
        t1.state = Some(proposed_state1);
    } else {
        t0.splat_state(&proposed_state0);
        t1.splat_state(&proposed_state1);

        t0.sampler.reject();
        t1.sampler.reject();

        std::mem::swap(&mut t0.sampler, &mut t1.sampler);
    }
}

fn replica_exchange_safe<F>(t0: &mut Tile, t1: &mut Tile, large_prob: f32, technique: F)
where
    F: Fn((u32, u32), &mut dyn Sampler) -> Color,
{
    if t0.state.is_none() && t1.state.is_none() {
        chain_non_init(t0, &technique);
        chain_non_init(t1, &technique);
    } else if t0.state.is_some() && t1.state.is_some() {
        replica_exchange(t0, t1, technique);
    } else {
        if BORROW {
            // One is initialized and another one not
            let (init, non_init) = if t1.state.is_some() {
                (t1, t0)
            } else {
                (t0, t1)
            };
            assert!(init.state.is_some());
            assert!(non_init.state.is_none());

            // For the non init, we will borrow the state of the init
            // one and try to generate a path
            // FIXME: This is dangerous code as a bad copy of the internal
            //  states can leads to bugs
            non_init.sampler.values = init.sampler.values.clone();
            non_init.sampler.time = init.sampler.time;
            non_init.sampler.time_large = init.sampler.time_large;
            non_init.sampler.large_step = false;
            {
                let state = non_init.generate_state(&technique);
                // Always accept (as it does not matter)
                non_init.sampler.accept();
                if state.tf != 0.0 {
                    non_init.state = Some(state);
                }
            }

            // For the other sampler, we just continue :)
            independent_mcmc(init, large_prob, technique);
        } else {
            independent_mcmc_safe(t0, large_prob, &technique);
            independent_mcmc_safe(t1, large_prob, &technique);
        }
    }
}

pub trait Reconstruction: Send + Sync {
    fn reconstruction(&self, tiles: &Vec<Tile>, img_size: Vector2<u32>) -> BufferCollection;
}
pub struct ReconstructionNaive;
impl Reconstruction for ReconstructionNaive {
    fn reconstruction(&self, tiles: &Vec<Tile>, img_size: Vector2<u32>) -> BufferCollection {
        let buffer_names = vec!["primal".to_string()];
        let mut img = BufferCollection::new(Point2::new(0, 0), img_size, &buffer_names);

        let mut accum = vec![Color::zero(); (img_size.x * img_size.y) as usize];
        let mut sample_count = vec![0; (img_size.x * img_size.y) as usize];
        for t in tiles {
            // Skip if no normalization
            if t.b == 0.0 {
                continue;
            }
            assert!(t.nb_uniform > 0);
            let norm = t.b / t.nb_uniform as f32;
            if t.nb_samples > 0 {
                for p in &t.pixels {
                    let i = (p.p.y * img_size.x + p.p.x) as usize;
                    accum[i] += p.value * norm;
                    sample_count[i] += t.nb_samples;
                }
            }
        }

        // Splat inside the bitmap
        for y in 0..img_size.y {
            for x in 0..img_size.x {
                let i = (y * img_size.x + x) as usize;
                if sample_count[i] > 0 {
                    img.accumulate(
                        Point2::new(x, y),
                        accum[i] / sample_count[i] as f32,
                        &buffer_names[0],
                    );
                }
            }
        }

        img
    }
}
pub struct ReconstructionIRLS {
    pub irls_iter: usize,
    pub internal_iter: usize,
    pub alpha: f32,
}
mod irls {
    // These structure is to store the tile information
    // efficiently, avoiding recomputing operation
    pub struct TileSumStats {
        pub mcmc: f32,
        pub mc: f32,
    }
    pub struct CacheTile {
        pub center: f32,
        pub left: f32,
        pub right: f32,
        pub top: f32,
        pub down: f32,
    }
    impl Default for CacheTile {
        fn default() -> Self {
            Self {
                center: 0.0,
                left: 0.0,
                right: 0.0,
                top: 0.0,
                down: 0.0,
            }
        }
    }

    // This is a structure to helps
    // to compute stats for reconstruction
    const CUSTOM_W: bool = false;
    pub trait Op {
        // p = (b, v, w)
        fn update(&mut self, p1: (f32, f32, f32), p2: (f32, f32, f32));
        fn value(self, curr_b: f32) -> f32;
    }
    pub struct ReconsOp {
        pub force: f32,
        pub pos: f32,
    }
    impl Default for ReconsOp {
        fn default() -> Self {
            Self {
                force: 0.0,
                pos: 0.0,
            }
        }
    }
    impl Op for ReconsOp {
        fn update(&mut self, (b1, v1, w1): (f32, f32, f32), (b2, v2, w2): (f32, f32, f32)) {
            let w = w1.min(w2);
            let e1 = v1 * b1;
            let e2 = v2 * b2;
            let f = 0.5 * (e1 - e2);
            let align = if crate::integrators::mcmc::smcmc::BORROW {
                v1 != 0.0 && v2 != 0.0
            } else {
                e1 != 0.0 && e2 != 0.0
            };
            if align {
                let wc = if CUSTOM_W {
                    (v1.min(v2) / v1.max(v2)).powi(2)
                } else {
                    1.0
                };
                self.force += w * wc * f;
                self.pos += w * wc * v1;
            }
        }

        fn value(self, curr_b: f32) -> f32 {
            if self.pos == 0.0 {
                curr_b
            } else {
                let b = curr_b - self.force / self.pos;
                if b.is_finite() {
                    b
                } else {
                    curr_b
                }
            }
        }
    }
    pub struct ErrorOp {
        pub error: f32,
    }
    impl Default for ErrorOp {
        fn default() -> Self {
            Self { error: 0.0 }
        }
    }
    impl Op for ErrorOp {
        fn update(&mut self, (b1, v1, _w1): (f32, f32, f32), (b2, v2, _w2): (f32, f32, f32)) {
            let e1 = v1 * b1;
            let e2 = v2 * b2;
            let f = 0.5 * (e1 - e2);
            let align = if crate::integrators::mcmc::smcmc::BORROW {
                v1 != 0.0 && v2 != 0.0
            } else {
                e1 != 0.0 && e2 != 0.0
            };
            if align {
                let w = if CUSTOM_W {
                    (v1.min(v2) / v1.max(v2)).powi(2)
                } else {
                    1.0
                };
                self.error += f.abs() * w;
            }
        }
        fn value(self, _curr_b: f32) -> f32 {
            self.error
        }
    }
}

impl ReconstructionIRLS {
    pub fn apply_op<F>(
        &self,
        pixel_order: &Vec<Point2<u32>>,
        sums: &Vec<irls::TileSumStats>,
        cache: &Vec<irls::CacheTile>,
        b: &Vec<f32>,
        w: &Vec<f32>,
        img_size: &Vector2<u32>,
    ) -> Vec<f32>
    where
        F: irls::Op + Default,
    {
        pixel_order
            .par_iter()
            .map(|p| {
                let curr_id = (p.y * img_size.x + p.x) as usize;
                let curr_cache = &cache[curr_id];
                let curr_sums = &sums[curr_id];
                let curr_b = b[curr_id];
                let curr_w = w[curr_id];

                // If no MCMC estimates, just finish to process
                // this tile!
                if curr_sums.mcmc == 0.0 {
                    return curr_b;
                }

                // Build the object where we will accumulate stats
                let mut res = F::default();

                // Regularisation factor
                res.update(
                    (curr_b, curr_sums.mcmc, self.alpha * curr_w),
                    (1.0, curr_sums.mc, self.alpha * curr_w),
                );

                // +X, -X, +Y, -Y
                if p.x != 0 {
                    // Overlap center and left
                    let next_id = curr_id - 1;
                    let next_cache = &cache[next_id];
                    let next_b = b[next_id];
                    let next_w = w[next_id];

                    res.update(
                        (curr_b, curr_cache.center, curr_w),
                        (next_b, next_cache.right, next_w),
                    );
                    res.update(
                        (curr_b, curr_cache.left, curr_w),
                        (next_b, next_cache.center, next_w),
                    );
                }
                if p.x != img_size.x - 1 {
                    // Overlap center and right
                    let next_id = curr_id + 1;
                    let next_cache = &cache[next_id];
                    let next_b = b[next_id];
                    let next_w = w[next_id];

                    res.update(
                        (curr_b, curr_cache.center, curr_w),
                        (next_b, next_cache.left, next_w),
                    );
                    res.update(
                        (curr_b, curr_cache.right, curr_w),
                        (next_b, next_cache.center, next_w),
                    );
                }
                if p.y != 0 {
                    // Overlap center and top
                    let next_id = curr_id - img_size.x as usize;
                    let next_cache = &cache[next_id];
                    let next_b = b[next_id];
                    let next_w = w[next_id];

                    res.update(
                        (curr_b, curr_cache.center, curr_w),
                        (next_b, next_cache.down, next_w),
                    );
                    res.update(
                        (curr_b, curr_cache.top, curr_w),
                        (next_b, next_cache.center, next_w),
                    );
                }
                if p.y != img_size.y - 1 {
                    // Overlap center and down
                    let next_id = curr_id + img_size.x as usize;
                    let next_cache = &cache[next_id];
                    let next_b = b[next_id];
                    let next_w = w[next_id];

                    res.update(
                        (curr_b, curr_cache.center, curr_w),
                        (next_b, next_cache.top, next_w),
                    );
                    res.update(
                        (curr_b, curr_cache.down, curr_w),
                        (next_b, next_cache.center, next_w),
                    );
                }

                // Diagonals
                if p.x != 0 && p.y != 0 {
                    // Overlap top and left
                    let next_id = curr_id - 1 - img_size.x as usize;
                    let next_cache = &cache[next_id];
                    let next_b = b[next_id];
                    let next_w = w[next_id];

                    res.update(
                        (curr_b, curr_cache.top, curr_w),
                        (next_b, next_cache.right, next_w),
                    );
                    res.update(
                        (curr_b, curr_cache.left, curr_w),
                        (next_b, next_cache.down, next_w),
                    );
                }
                if p.x != img_size.x - 1 && p.y != img_size.y - 1 {
                    // Overlap bottom and right
                    let next_id = curr_id + 1 + img_size.x as usize;
                    let next_cache = &cache[next_id];
                    let next_b = b[next_id];
                    let next_w = w[next_id];

                    res.update(
                        (curr_b, curr_cache.down, curr_w),
                        (next_b, next_cache.left, next_w),
                    );
                    res.update(
                        (curr_b, curr_cache.right, curr_w),
                        (next_b, next_cache.top, next_w),
                    );
                }
                if p.x != 0 && p.y != img_size.y - 1 {
                    // Overlap bottom and left
                    let next_id = curr_id - 1 + img_size.x as usize;
                    let next_cache = &cache[next_id];
                    let next_b = b[next_id];
                    let next_w = w[next_id];

                    res.update(
                        (curr_b, curr_cache.down, curr_w),
                        (next_b, next_cache.right, next_w),
                    );
                    res.update(
                        (curr_b, curr_cache.left, curr_w),
                        (next_b, next_cache.top, next_w),
                    );
                }
                if p.x != img_size.x - 1 && p.y != 0 {
                    // Overlap top and right
                    let next_id = curr_id + 1 - img_size.x as usize;
                    let next_cache = &cache[next_id];
                    let next_b = b[next_id];
                    let next_w = w[next_id];

                    res.update(
                        (curr_b, curr_cache.top, curr_w),
                        (next_b, next_cache.left, next_w),
                    );
                    res.update(
                        (curr_b, curr_cache.right, curr_w),
                        (next_b, next_cache.down, next_w),
                    );
                }

                // Only one overlap
                if p.x > 1 {
                    // Overlap left-right
                    let next_id = curr_id - 2;
                    let next_cache = &cache[next_id];
                    let next_b = b[next_id];
                    let next_w = w[next_id];

                    res.update(
                        (curr_b, curr_cache.left, curr_w),
                        (next_b, next_cache.right, next_w),
                    );
                }
                if p.x < img_size.x - 2 {
                    // Overlap right-left
                    let next_id = curr_id + 2;
                    let next_cache = &cache[next_id];
                    let next_b = b[next_id];
                    let next_w = w[next_id];

                    res.update(
                        (curr_b, curr_cache.right, curr_w),
                        (next_b, next_cache.left, next_w),
                    );
                }
                if p.y > 1 {
                    // Overlap top-down
                    let next_id = curr_id - 2 * img_size.x as usize;
                    let next_cache = &cache[next_id];
                    let next_b = b[next_id];
                    let next_w = w[next_id];

                    res.update(
                        (curr_b, curr_cache.top, curr_w),
                        (next_b, next_cache.down, next_w),
                    );
                }
                if p.y < img_size.y - 2 {
                    // Overlap down-top
                    let next_id = curr_id + 2 * img_size.x as usize;
                    let next_cache = &cache[next_id];
                    let next_b = b[next_id];
                    let next_w = w[next_id];

                    res.update(
                        (curr_b, curr_cache.down, curr_w),
                        (next_b, next_cache.top, next_w),
                    );
                }

                res.value(curr_b)
            })
            .collect::<Vec<_>>()
    }

    // Note that the reconstruction is independent per wavelength
    // This improve reconstruction by removing color noise
    // This way is valid as color channels are correlated.
    pub fn weighted_reconstruction_channel<F>(
        &self,
        tiles: &Vec<Tile>,
        channel: F,
        img_size: Vector2<u32>,
    ) -> Vec<f32>
    where
        F: Fn(&Color) -> f32,
    {
        // Compute the MC estimates by combining all tiles estimates
        let mc_estimates = {
            let mut accum = vec![0.0; (img_size.x * img_size.y) as usize];
            let mut sample_count = vec![0; (img_size.x * img_size.y) as usize];
            for t in tiles {
                // Skip if no normalization
                if t.b == 0.0 {
                    continue;
                }
                assert!(t.nb_uniform > 0);
                for p in &t.pixels {
                    let i = (p.p.y * img_size.x + p.p.x) as usize;
                    accum[i] += channel(&p.value_mc);
                    sample_count[i] += t.nb_uniform;
                }
            }

            accum
                .iter()
                .zip(sample_count.iter())
                .map(|(a, i)| if *i == 0 { 0.0 } else { a / *i as f32 })
                .collect::<Vec<_>>()
        };

        // Collect tiles sums stats
        // these will be useful for to compute the regularisation term
        let sums = tiles
            .iter()
            .map(|t| {
                // For mc estimate, we uses the robust estimate
                // instead of the chain own estimates. This improve
                // the quality of the reconstruction as the regularisation term
                // is more robustly estimated.
                let mc = t
                    .pixels
                    .iter()
                    .map(|p| {
                        let i = p.p.y * img_size.x + p.p.x;
                        // TODO: Check MCMC estimates?
                        mc_estimates[i as usize]
                    })
                    .sum();
                let mcmc = t.pixels.iter().map(|p| channel(&p.value)).sum();
                // TODO: Normally, we should normalize it
                //  But it seems uncessary for the first implementation
                //  However, we need to be careful as different number of pixels
                //  will implies different regularisation "force"
                irls::TileSumStats { mcmc, mc }
            })
            .collect::<Vec<_>>();

        // Compute cache efficient reconstruction
        let cache = tiles
            .iter()
            .map(|t| {
                let c = t.center_pos();
                let mut cache = irls::CacheTile::default();
                for p in &t.pixels {
                    let x = c.x as i32 - p.p.x as i32;
                    let y = c.y as i32 - p.p.y as i32;
                    let v = channel(&p.value);
                    match (x, y) {
                        (0, 0) => cache.center = v,
                        (1, 0) => cache.left = v,
                        (-1, 0) => cache.right = v,
                        (0, -1) => cache.down = v,
                        (0, 1) => cache.top = v,
                        _ => panic!("Wrong tile mapping x={} y={} ({:?})", x, y, &t.pixels),
                    };
                }
                cache
            })
            .collect::<Vec<_>>();

        // Build the pixel indice pattern
        // This is usefull for the paralelisation
        let pixel_order = {
            let mut pixel_order = Vec::with_capacity((img_size.x * img_size.y) as usize);
            for y in 0..img_size.y {
                for x in 0..img_size.x {
                    pixel_order.push(Point2::new(x, y));
                }
            }
            pixel_order
        };

        // Do the optimization (iterative reweight LS)
        let mut w = vec![1.0; (img_size.x * img_size.y) as usize];
        // Get the normalization factor for each tiles
        let mut b = tiles
            .iter()
            .map(|t| match t.nb_uniform {
                0 => 0.0,
                v => t.b / v as f32,
            })
            .collect::<Vec<_>>();
        //let b0 = b.clone();
        for iter in 0..self.irls_iter {
            //b.iter_mut().zip(b0.iter()).for_each(|(b, b0)| *b = *b0);
            for _internal_iter in 0..self.internal_iter {
                // TODO: This allocation can be slow!
                //  it might be possible to pre-alloc b_next
                //  and zip the two vectors (or pixel_order store the results)
                let b_next =
                    self.apply_op::<irls::ReconsOp>(&pixel_order, &sums, &cache, &b, &w, &img_size);

                // Update inplace
                b.iter_mut()
                    .zip(b_next.into_iter())
                    .for_each(|(b, b_next)| {
                        assert!(b_next >= 0.0, "b_next negative {}", b_next);
                        *b = b_next;
                    });
            }

            // Compute weights
            let mut w_new =
                self.apply_op::<irls::ErrorOp>(&pixel_order, &sums, &cache, &b, &w, &img_size);
            for w in &mut w_new {
                *w = 1.0 / (*w + (0.05 * (0.5f32).powi(iter as i32)).max(0.0001))
            }

            // Compute normalization and update w
            let w_sum = w_new.iter().sum::<f32>();
            w.iter_mut().zip(w_new).for_each(|(w, w_new)| {
                *w = w_new * (img_size.x * img_size.y) as f32 / w_sum;
                assert!(w.is_finite());
            });
        }

        // FIXME: Check if we can make b stay the same
        //  for efficient reconstruction
        // b.iter_mut().zip(b0.iter()).for_each(|(b, b0)| *b = *b0);
        // for _internal_iter in 0..200 {
        //     let b_next = apply_op::<ReconsOp>(&pixel_order, &sums, &cache, &b, &w, &img_size);

        //     // Update inplace
        //     b.iter_mut()
        //         .zip(b_next.into_iter())
        //         .for_each(|(b, b_next)| {
        //             assert!(b_next >= 0.0, "b_next negative {}", b_next);
        //             *b = b_next;
        //         });
        // }

        b
    }
}
impl Reconstruction for ReconstructionIRLS {
    fn reconstruction(&self, tiles: &Vec<Tile>, img_size: Vector2<u32>) -> BufferCollection {
        let buffer_names = vec!["primal".to_string()];
        let mut img = BufferCollection::new(Point2::new(0, 0), img_size, &buffer_names);

        let b_r = self.weighted_reconstruction_channel(tiles, |c: &Color| -> f32 { c.r }, img_size);
        let b_g = self.weighted_reconstruction_channel(tiles, |c: &Color| -> f32 { c.g }, img_size);
        let b_b = self.weighted_reconstruction_channel(tiles, |c: &Color| -> f32 { c.b }, img_size);

        let mut accum = vec![Color::zero(); (img_size.x * img_size.y) as usize];
        let mut sample_count = vec![0; (img_size.x * img_size.y) as usize];
        for (i, t) in tiles.iter().enumerate() {
            // Skip if no normalization
            if b_r[i] == 0.0 && b_g[i] == 0.0 && b_b[i] == 0.0 {
                continue;
            }
            if t.nb_samples > 0 {
                for p in &t.pixels {
                    let i = (p.p.y * img_size.x + p.p.x) as usize;
                    accum[i] +=
                        Color::new(p.value.r * b_r[i], p.value.g * b_g[i], p.value.b * b_b[i]);
                    sample_count[i] += t.nb_samples;
                }
            }
        }

        // Splat inside the bitmap
        for y in 0..img_size.y {
            for x in 0..img_size.x {
                let i = (y * img_size.x + x) as usize;
                if sample_count[i] > 0 {
                    img.accumulate(
                        Point2::new(x, y),
                        accum[i] / sample_count[i] as f32,
                        &buffer_names[0],
                    );
                }
            }
        }

        img
    }
}

pub trait Initialization: Send + Sync {
    fn init(
        &self,
        img_size: &Vector2<u32>,
        accel: &dyn Acceleration,
        scene: &Scene,
        int: &dyn IntegratorMC,
        pool: &rayon::ThreadPool,
    ) -> Vec<Tile>;
}
pub struct IndependentInit {
    pub nb_spp: usize,
}
impl Initialization for IndependentInit {
    fn init(
        &self,
        img_size: &Vector2<u32>,
        accel: &dyn Acceleration,
        scene: &Scene,
        int: &dyn IntegratorMC,
        pool: &rayon::ThreadPool,
    ) -> Vec<Tile> {
        let mut chains = Vec::with_capacity((img_size.x * img_size.y) as usize);
        for y in 0..img_size.y {
            for x in 0..img_size.x {
                chains.push(Tile::new(Point2::new(x, y), img_size));
            }
        }

        let nb_initialized_total = Mutex::new(0);
        pool.install(|| {
            chains
                .par_chunks_mut(img_size.y as usize)
                .for_each(|tiles| {
                    let emitters = scene.emitters_sampler();
                    let technique = |p: (u32, u32), s: &mut dyn Sampler| -> Color {
                        int.compute_pixel((p.0, p.1), accel, scene, s, &emitters)
                    };
                    let mut nb_initialized = 0;
                    for tile in &mut tiles[..] {
                        // Initialize the state from uniform sampling
                        // This is the most naive approach.
                        // We should uses a global chain to initialize the states
                        tile.sampler.large_step = true;
                        // Naively estimate the normalization factor
                        tile.b = 0.0;
                        tile.nb_uniform = 0;
                        for p in &mut tile.pixels {
                            p.value_mc = Color::zero();
                        }
                        for _ in 0..self.nb_spp {
                            chain_non_init(tile, technique);
                            if tile.state.is_some() {
                                nb_initialized += 1;
                                break; // Finish for now
                            }
                        }
                    }
                    *nb_initialized_total.lock().unwrap() += nb_initialized;
                });
        });
        let nb_initialized_total = nb_initialized_total.into_inner().unwrap();
        info!(
            "Number of tile initialized: {} %",
            100.0 * nb_initialized_total as f32 / (img_size.x * img_size.y) as f32
        );
        chains
    }
}
pub struct MCMCInit {
    pub spp_mc: usize,
    pub spp_mcmc: usize,
    pub chain_length: usize,
}
impl Initialization for MCMCInit {
    fn init(
        &self,
        img_size: &Vector2<u32>,
        accel: &dyn Acceleration,
        scene: &Scene,
        int: &dyn IntegratorMC,
        pool: &rayon::ThreadPool,
    ) -> Vec<Tile> {
        // Create protected tiles
        let mut entries = Vec::with_capacity((img_size.x * img_size.y) as usize);
        pub struct TileEntry {
            nb_visit: usize,
            tile: Tile,
        }
        for y in 0..img_size.y {
            for x in 0..img_size.x {
                entries.push(Mutex::new(TileEntry {
                    nb_visit: 0,
                    tile: Tile::new(Point2::new(x, y), img_size),
                }));
            }
        }

        pool.install(|| {
            // Initial seed creation
            let seeds: Vec<(rand::rngs::SmallRng, f32, Point2<u32>)> = entries
                .par_chunks_mut(img_size.y as usize)
                .map(|entries| {
                    let emitters = scene.emitters_sampler();
                    let technique = |p: (u32, u32), s: &mut dyn Sampler| -> Color {
                        int.compute_pixel((p.0, p.1), accel, scene, s, &emitters)
                    };
                    let mut seeds = vec![];
                    for entry in &mut entries[..] {
                        let mut entry = entry.lock().unwrap();
                        entry.tile.sampler.large_step = true;
                        // Naively estimate the normalization factor
                        entry.tile.b = 0.0;
                        entry.tile.nb_uniform = 0;
                        for p in &mut entry.tile.pixels {
                            p.value_mc = Color::zero();
                        }
                        for _ in 0..self.spp_mc {
                            let current_seed = entry.tile.sampler.rnd.clone();
                            entry.tile.sampler.large_step = true;
                            let state = entry.tile.generate_state(technique);
                            // Always accept (as it does not matter)
                            entry.tile.sampler.accept();
                            // Estimate normalization factor
                            entry.tile.splat_state_uni(&state);
                            if state.tf != 0.0 {
                                seeds.push((current_seed, state.tf, entry.tile.center_pos()));
                            }
                        }
                    }
                    seeds
                })
                .flatten()
                .collect::<Vec<_>>();

            // Construct the distribution to pick the seeds
            let mut cdf = Distribution1DConstruct::new(seeds.len());
            for s in &seeds {
                cdf.add(s.1);
            }
            let cdf = cdf.normalize();
            if cdf.normalization == 0.0 {
                panic!("Normalization is 0, impossible to continue");
            }

            // Floor but it is fine
            let nb_chains =
                ((img_size.x * img_size.y) as usize * self.spp_mcmc) / self.chain_length;
            assert!(nb_chains != 0);
            let workers = (0..nb_chains)
                .map(|id_chain| (id_chain, Tile::new(Point2::new(0, 0), img_size)))
                .collect::<Vec<_>>();
            workers
                .into_par_iter()
                .for_each(|(chain_id, mut dumy_tile)| {
                    let img_mut = MutatorKelemen::default();
                    let emitters = scene.emitters_sampler();
                    let technique = |p: (u32, u32), s: &mut dyn Sampler| -> Color {
                        int.compute_pixel((p.0, p.1), accel, scene, s, &emitters)
                    };

                    // Pick one with stratified sampling
                    let v_id = (chain_id as f32 + dumy_tile.sampler.rand()) / nb_chains as f32;
                    let seed = &seeds[cdf.sample(v_id)];

                    // Save the rng for later
                    let ori_rng = dumy_tile.sampler.rnd.clone();

                    // Regenerate the path
                    dumy_tile.reallocate(seed.2, img_size);
                    dumy_tile.sampler.rnd = seed.0.clone();
                    dumy_tile.sampler.large_step = true;
                    let mut state = dumy_tile.generate_state(technique);
                    dumy_tile.sampler.accept();
                    if state.tf != seed.1 {
                        panic!("Different TF: {} != {}", state.tf, seed.1);
                    }

                    // Reput the original random gen
                    // to ensure that the chains will be not sync
                    // if picking the same seed
                    dumy_tile.sampler.rnd = ori_rng;

                    for _ in 0..self.chain_length {
                        // Start to do the MCMC stuff
                        // As the chain is already init, we will:
                        //  1) Check if we want to initialize the active tile
                        //  2) Mutate the dumy_tile to visit other tile
                        let pos_dumy = dumy_tile.center_pos();
                        let id_tile = pos_dumy.y * img_size.x + pos_dumy.x;

                        // Now we require exclusive access to the active tile
                        {
                            let mut entry = entries[id_tile as usize].lock().unwrap();

                            // Do reservoir sampling style to know if we want to duplicate the state
                            entry.nb_visit += 1;
                            let replace =
                                (dumy_tile.sampler.rand() * entry.nb_visit as f32) as usize == 0;

                            // If we decide to replace the active tile state
                            if replace {
                                // FIXME: This is dangerous code as a bad copy of the internal
                                //  states can leads to bugs

                                // Manipulate the sampler
                                entry.tile.sampler.values = dumy_tile.sampler.values.clone();
                                entry.tile.sampler.time = dumy_tile.sampler.time;
                                entry.tile.sampler.time_large = dumy_tile.sampler.time_large;
                                entry.tile.sampler.large_step = false;

                                // We do not need to retrace the path as we are
                                // on the same tile.
                                entry.tile.state = Some(state.clone());
                            }
                        }

                        // Decide the new tile to visit
                        // where we force to change to a new tile
                        let mut new_pos_x = (img_mut.mutate(
                            (pos_dumy.x as f32 + 0.5) / img_size.x as f32,
                            dumy_tile.sampler.rand(),
                            0,
                        ) * img_size.x as f32) as u32;
                        let mut new_pos_y = (img_mut.mutate(
                            (pos_dumy.y as f32 + 0.5) / img_size.y as f32,
                            dumy_tile.sampler.rand(),
                            1,
                        ) * img_size.y as f32) as u32;
                        while new_pos_x == pos_dumy.x && new_pos_y == pos_dumy.y {
                            new_pos_x = (img_mut.mutate(
                                (pos_dumy.x as f32 + 0.5) / img_size.x as f32,
                                dumy_tile.sampler.rand(),
                                0,
                            ) * img_size.x as f32) as u32;
                            new_pos_y = (img_mut.mutate(
                                (pos_dumy.y as f32 + 0.5) / img_size.y as f32,
                                dumy_tile.sampler.rand(),
                                1,
                            ) * img_size.y as f32) as u32;
                        }

                        // We reallocate position of the dumy
                        dumy_tile.reallocate(Point2::new(new_pos_x, new_pos_y), img_size);

                        // We generate a propose state in the new tile
                        dumy_tile.sampler.large_step = false;
                        let proposed_state = dumy_tile.generate_state(technique);
                        let accept_prob = (proposed_state.tf / state.tf).min(1.0);

                        // MH step
                        let accepted = accept_prob > dumy_tile.sampler.rand();
                        if accepted {
                            dumy_tile.sampler.accept();
                            state = proposed_state;
                        } else {
                            dumy_tile.sampler.reject();
                            // We need to revert back the position change
                            dumy_tile.reallocate(pos_dumy, img_size);
                        }
                    }
                });
        });

        entries
            .into_iter()
            .map(|e| e.into_inner().unwrap().tile)
            .collect()
    }
}

pub struct StratifiedMCMC {
    pub integrator: Box<dyn IntegratorMC>,
    // We use the option here to inform the integrator
    // if we never created the chains
    pub chains: Option<Vec<Tile>>,
    /// The probability to do large step
    pub large_prob: f32,
    /// The image reconstruction algorithm
    pub recons: Box<dyn Reconstruction>,
    /// How to initialize the chains
    pub init: Box<dyn Initialization>,
}
impl Integrator for StratifiedMCMC {
    fn averaging(&self) -> bool {
        // The reconstructed image have all the average information
        // from previous passes
        false
    }

    fn compute(
        &mut self,
        _sampler: &mut dyn Sampler,
        accel: &dyn Acceleration,
        scene: &Scene,
    ) -> BufferCollection {
        // Image size
        let img_size = scene.camera.size();
        // Thread pool
        let pool = generate_pool(scene);

        if self.chains.is_none() {
            let start = Instant::now();
            self.chains =
                Some(
                    self.init
                        .init(img_size, accel, scene, self.integrator.as_ref(), &pool),
                );
            info!("Initialisation Elapsed: {:?}", start.elapsed());
        };

        ///////////// Compute the rendering (with the number of samples)
        info!("Rendering...");
        let start = Instant::now();
        let progress_bar = Mutex::new(ProgressBar::new(
            img_size.y as u64 * scene.nb_samples as u64,
        ));

        pool.install(|| {
            // This is a bit unfortunate but this line is necessary
            // to overcome borrow and mut borrow from self (borrow-checker)
            let (chains, int, large_prob) = (
                self.chains.as_mut().unwrap(),
                self.integrator.as_ref(),
                self.large_prob,
            );

            #[derive(Debug)]
            enum State {
                MCMC(u8),
                Horizontal(bool),
                Vertical(bool),
            }

            // TODO: Should change the state to random [0, 3]
            // (this will be important if the number of SPP is not 8 multiple)
            let mut state = State::MCMC(0);

            let emitters = scene.emitters_sampler();
            let technique = |p: (u32, u32), s: &mut dyn Sampler| -> Color {
                int.compute_pixel((p.0, p.1), accel, scene, s, &emitters)
            };

            // For now, it is good assumptions
            assert!(img_size.x % 2 == 0);
            assert!(img_size.y % 2 == 0);

            for _ in 0..scene.nb_samples {
                // Do one step of sampling
                match state {
                    State::MCMC(_) => {
                        chains
                            .par_chunks_mut(img_size.x as usize)
                            .for_each(|tiles| {
                                for t in &mut tiles[..] {
                                    independent_mcmc_safe(t, large_prob, technique);
                                }

                                {
                                    progress_bar.lock().unwrap().inc();
                                }
                            });
                    }
                    State::Horizontal(b) => {
                        // The paralelisation can be done line to line
                        chains
                            .par_chunks_mut(img_size.x as usize)
                            .for_each(|tiles| {
                                let offset = if b { 1 } else { 0 };
                                for t in tiles[offset..].chunks_exact_mut(2) {
                                    match t {
                                        [t0, t1] => {
                                            replica_exchange_safe(t0, t1, large_prob, technique)
                                        }
                                        _ => panic!("No pair slices"),
                                    }
                                }

                                // If we doing the offset, we need to complete
                                if b {
                                    independent_mcmc_safe(&mut tiles[0], large_prob, technique);
                                    independent_mcmc_safe(
                                        &mut tiles[tiles.len() - 1],
                                        large_prob,
                                        technique,
                                    );
                                }

                                {
                                    progress_bar.lock().unwrap().inc();
                                }
                            });
                    }
                    State::Vertical(b) => {
                        // Here is a bit more involved as we need to two lines to do the exchange
                        // This will be fine, but if there is stride the most top and bottom line
                        // need to be process independently
                        let offset = if b { img_size.x as usize } else { 0 };
                        chains[offset..]
                            .par_chunks_exact_mut(2 * img_size.x as usize)
                            .for_each(|tiles| {
                                let (t_line0, t_line1) = tiles.split_at_mut(img_size.x as usize);
                                for (t0, t1) in t_line0.iter_mut().zip(t_line1.iter_mut()) {
                                    replica_exchange_safe(t0, t1, large_prob, technique);
                                }

                                {
                                    progress_bar.lock().unwrap().inc();
                                    progress_bar.lock().unwrap().inc();
                                }
                            });

                        if b {
                            // Need to process the two image slice independently
                            chains[..offset].par_iter_mut().for_each(|mut t| {
                                independent_mcmc_safe(&mut t, large_prob, technique);
                            });
                            let last_line = (img_size.x - 1) * img_size.y;
                            chains[(last_line as usize)..]
                                .par_iter_mut()
                                .for_each(|mut t| {
                                    independent_mcmc_safe(&mut t, large_prob, technique);
                                });
                            // Update progress bar :)
                            progress_bar.lock().unwrap().inc();
                            progress_bar.lock().unwrap().inc();
                        }
                    }
                };

                // Change the state
                state = match state {
                    State::MCMC(v) => match v {
                        0 => State::Horizontal(false),
                        1 => State::Vertical(false),
                        2 => State::Horizontal(true),
                        3 => State::Vertical(true),
                        _ => panic!("Wrong MCMC state value {}", v),
                    },
                    State::Horizontal(b) => {
                        if b {
                            State::MCMC(3)
                        } else {
                            State::MCMC(1)
                        }
                    }
                    State::Vertical(b) => {
                        if b {
                            State::MCMC(0)
                        } else {
                            State::MCMC(2)
                        }
                    }
                };
            }
        });

        let elapsed = start.elapsed();
        info!("Elapsed: {:?}", elapsed,);

        // Finish and fill the normal buffer
        // naive_reconstruction(self.chains.as_ref().unwrap(), *img_size)
        self.recons
            .reconstruction(self.chains.as_ref().unwrap(), *img_size)
    }
}
