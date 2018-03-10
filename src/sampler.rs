// For random number
use rand;
use rand::distributions::{IndependentSample, Range};

// FIXME: This code is not used for now.
// FIXME: Found a way to make compatible with multi-threading
pub trait Sampler {
    fn next(& mut self) -> f32;
    fn next2d(& mut self) -> (f32, f32);
}

pub struct IndepSampler {
    rnd : rand::StdRng,
    dist : Range<f32>,
}

impl Sampler for IndepSampler {
    fn next(& mut self) -> f32 {
        self.dist.ind_sample(& mut self.rnd)
    }
    fn next2d(& mut self) -> (f32,f32) {
        (self.dist.ind_sample(& mut self.rnd),self.dist.ind_sample(& mut self.rnd))
    }
}

// TODO: Implement clone
// TODO: be able to use
impl IndepSampler {
    pub fn new() -> IndepSampler {
        IndepSampler {
            rnd : rand::StdRng::new().unwrap(),
            dist : Range::new(0.0, 1.0),
        }
    }
}