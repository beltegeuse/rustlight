use crate::integrators::BufferCollection;
use crate::math::{Distribution1D, Distribution1DConstruct};
use crate::samplers::*;
use crate::structure::*;
use cgmath::Point2;
use rand::rngs::SmallRng;

#[derive(Clone)]
pub struct MCMCState {
    pub values: Vec<(Color, Point2<u32>)>,
    pub tf: f32,
    pub weight: f32,
}

impl MCMCState {
    pub fn empty() -> MCMCState {
        MCMCState {
            values: vec![],
            tf: 0.0,
            weight: 0.0,
        }
    }
    pub fn new(c: Color, p: Point2<u32>) -> MCMCState {
        MCMCState {
            values: vec![(c, p)],
            tf: (c.r + c.g + c.b) / 3.0,
            weight: 0.0,
        }
    }

    pub fn append(&mut self, c: Color, pos: Point2<u32>) {
        self.tf += (c.r + c.g + c.b) / 3.0;
        self.values.push((c, pos));
    }

    pub fn append_with_tf(&mut self, c: Color, tf: f32, pos: Point2<u32>) {
        self.tf += tf;
        self.values.push((c, pos));
    }

    pub fn accumulate(&mut self, img: &mut BufferCollection, buffer_name: &str) {
        let w = self.weight / self.tf;
        for (c, pos) in &self.values {
            img.accumulate(*pos, (*c) * w, buffer_name);
        }
        self.weight = 0.0;
    }

    pub fn accumulate_weight(&mut self, img: &mut BufferCollection, buffer_name: &str, w: f32) {
        let w = (self.weight / self.tf) * w;
        for (c, pos) in &self.values {
            img.accumulate(*pos, (*c) * w, buffer_name);
        }
        self.weight = 0.0;
    }

    pub fn accumulate_bitmap(&mut self, img: &mut Bitmap) {
        let w = self.weight / self.tf;
        for (c, pos) in &self.values {
            img.accumulate(*pos, (*c) * w);
        }
        self.weight = 0.0;
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
    let b = (0..nb_samples)
        .map(|_| {
            let current_seed = sampler.rnd.clone();
            let state = routine(&mut sampler);
            if state.tf > 0.0 {
                seeds.push((state.tf, current_seed));
            }
            state.tf
        })
        .sum::<f32>()
        / nb_samples as f32;
    if b == 0.0 {
        panic!("Normalization is 0, impossible to continue");
    }

    let mut cdf = Distribution1DConstruct::new(seeds.len());
    for s in &seeds {
        cdf.add(s.0);
    }
    let cdf = cdf.normalize();
    (b, seeds, cdf)
}

fn average_lum<F>(nb_samples: usize, routine: F) -> f32
where
    F: Fn(&mut dyn Sampler) -> MCMCState,
{
    // TODO: Here we do not need to change the way to sample the image space
    //  As there is no burning period implemented.
    let mut sampler = crate::samplers::independent::IndependentSampler::default();
    (0..nb_samples)
        .map(|_| {
            let state = routine(&mut sampler);
            state.tf
        })
        .sum::<f32>()
        / nb_samples as f32
}

pub mod erpt;
pub mod pssmlt;
pub mod smcmc;
