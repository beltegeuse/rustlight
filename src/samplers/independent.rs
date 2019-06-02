use crate::samplers::*;
use cgmath::Point2;
use rand::prelude::*;

pub struct IndependentSampler {
    rnd: StdRng,
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
}

impl Default for IndependentSampler {
    fn default() -> IndependentSampler {
        IndependentSampler {
            rnd: rand::rngs::StdRng::from_rng(thread_rng()).unwrap()
        }
    }
}
