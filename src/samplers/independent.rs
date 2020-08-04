use crate::samplers::*;
use cgmath::Point2;
use rand::prelude::*;

pub struct IndependentSampler {
    pub rnd: rand::rngs::SmallRng,
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
    fn clone_box(&mut self) -> Box<dyn Sampler> {
        Box::new(IndependentSampler {
            rnd: rand::rngs::SmallRng::seed_from_u64(self.rnd.next_u64()),
        })
    }

    fn next_sample(&mut self) {}
    fn next_pixel(&mut self, _: Point2<u32>) {}
}

impl Default for IndependentSampler {
    fn default() -> Self {
        IndependentSampler {
            rnd: rand::rngs::SmallRng::from_seed(random()),
        }
    }
}
