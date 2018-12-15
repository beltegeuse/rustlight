use crate::samplers::*;
use cgmath::Point2;
use rand;
use rand::distributions::{IndependentSample, Range};

pub struct IndependentSampler {
    rnd: rand::StdRng,
    dist: Range<f32>,
}

impl Sampler for IndependentSampler {
    fn next(&mut self) -> f32 {
        self.dist.ind_sample(&mut self.rnd)
    }
    fn next2d(&mut self) -> Point2<f32> {
        let x = self.dist.ind_sample(&mut self.rnd);
        let y = self.dist.ind_sample(&mut self.rnd);
        Point2::new(x, y)
    }
}

impl Default for IndependentSampler {
    fn default() -> IndependentSampler {
        IndependentSampler {
            rnd: rand::StdRng::new().unwrap(),
            dist: Range::new(0.0, 1.0),
        }
    }
}
