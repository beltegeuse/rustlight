use crate::samplers::*;
use cgmath::Point2;
use rand::prelude::*;

pub struct IndependentSampler {
    rnd: rand::rngs::SmallRng,
}

impl Sampler for IndependentSampler {
    fn next(&mut self) -> f32 {
        self.rnd.gen()
    }
    fn next2d(&mut self) -> Point2<f32> {
        let x = self.rnd.gen();
        let y = self.rnd.gen();
        Point2::new(x, y)
    }
    fn next_u64(&mut self) -> u64 {
        self.rnd.next_u64()
    }
    fn clone(&mut self) -> Box<dyn Sampler> {
        Box::new(IndependentSampler::from_seed(self.next_u64()))
    }
}

impl Default for IndependentSampler {
    fn default() -> Self {
        IndependentSampler::from_seed(random())
    }
}

impl IndependentSampler {
    pub fn from_seed(seed: u64) -> Self {
        IndependentSampler {
            rnd: rand::rngs::SmallRng::seed_from_u64(seed),
        }
    }
}
