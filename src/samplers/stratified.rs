use crate::clamp;
use crate::samplers::*;
use cgmath::Point2;
use rand::prelude::*;
use rand::rngs::SmallRng;

pub struct StratifiedSampler {
    // Indices
    pub index_1d: usize,
    pub index_2d: usize,
    pub index_sample: usize,
    // Parameters
    pub nb_dim: usize,     //< Number of dimension where using stratified samplers
    pub nb_samples: usize, //< Number of samples requested
    // Array and random source
    pub samples_1d: Vec<Vec<f32>>,
    pub samples_2d: Vec<Vec<Point2<f32>>>,
    pub rng: SmallRng,
}

impl StratifiedSampler {
    pub fn create(mut nb_samples: usize, nb_dim: usize) -> Self {
        let mut power_of4 = 1;
        while power_of4 < nb_samples {
            power_of4 *= 4;
        }
        if nb_samples != power_of4 {
            warn!("{} is not 4 multiple (increase count to {}), the stratified sampler will be less efficient!", nb_samples, power_of4);
            nb_samples = power_of4;
        }

        StratifiedSampler {
            index_1d: 0,
            index_2d: 0,
            index_sample: 0,
            nb_dim,
            nb_samples,
            samples_1d: vec![vec![0.0; nb_samples]; nb_dim],
            samples_2d: vec![vec![Point2::new(0.0, 0.0); nb_samples]; nb_dim],
            rng: SmallRng::from_seed(random()),
        }
    }
    fn regenerate(&mut self) {
        assert_eq!(self.nb_samples % 4, 0);
        // Refill and shuffle 1D vectors
        {
            let size = 1.0 / self.nb_samples as f32;
            for slice_1d in &mut self.samples_1d {
                for (i, s) in slice_1d.iter_mut().enumerate() {
                    let v = i as f32 * size + self.rng.gen::<f32>() * size;
                    // assert!(v >= 0.0 && v < 1.0, "1D stratified sample out of bound {}" , v);
                    *s = clamp(v, 0.0, 1.0 - f32::EPSILON);
                }
                slice_1d.shuffle(&mut self.rng);
            }
        }
        // Refill and shuffle 2D vectors
        {
            let samples_per_dim = (1.0 + self.nb_samples as f32).sqrt() as usize;
            let size = 1.0 / samples_per_dim as f32;
            for slice_2d in &mut self.samples_2d {
                for (x, slice_1d) in slice_2d.chunks_mut(samples_per_dim).enumerate() {
                    for (y, s) in slice_1d.iter_mut().enumerate() {
                        let vx = x as f32 * size + self.rng.gen::<f32>() * size;
                        // assert!(vx >= 0.0 && vx < 1.0, "2D stratified sample out of bound {} (x)" , vx);
                        let vy = y as f32 * size + self.rng.gen::<f32>() * size;
                        // assert!(vy >= 0.0 && vy < 1.0, "2D stratified sample out of bound {} (y)" , vy);
                        *s = Point2::new(
                            clamp(vx, 0.0, 1.0 - f32::EPSILON),
                            clamp(vy, 0.0, 1.0 - f32::EPSILON),
                        );
                    }
                }
                slice_2d.shuffle(&mut self.rng);
            }
        }
        self.index_1d = 0;
        self.index_2d = 0;
        self.index_sample = 0;
    }
}

impl Sampler for StratifiedSampler {
    fn next(&mut self) -> f32 {
        if self.index_1d >= self.samples_1d.len() {
            self.rng.gen()
        } else {
            assert!(self.index_sample < self.nb_samples);
            let v = self.samples_1d[self.index_1d][self.index_sample];
            self.index_1d += 1;
            v
        }
    }
    fn next2d(&mut self) -> Point2<f32> {
        if self.index_2d >= self.samples_2d.len() {
            Point2::new(self.rng.gen(), self.rng.gen())
        } else {
            assert!(self.index_sample < self.nb_samples);
            let v = self.samples_2d[self.index_2d][self.index_sample];
            self.index_2d += 1;
            v
        }
    }
    fn clone_box(&mut self) -> Box<dyn Sampler> {
        Box::new(StratifiedSampler {
            index_1d: 0,
            index_2d: 0,
            index_sample: 0,
            nb_dim: self.nb_dim,
            nb_samples: self.nb_samples,
            samples_1d: vec![vec![0.0; self.nb_samples]; self.nb_dim],
            samples_2d: vec![vec![Point2::new(0.0, 0.0); self.nb_samples]; self.nb_dim],
            rng: SmallRng::from_seed(random()),
        })
    }

    fn next_sample(&mut self) {
        self.index_1d = 0;
        self.index_2d = 0;
        self.index_sample += 1;
    }
    fn next_pixel(&mut self, _: Point2<u32>) {
        self.regenerate();
    }
}
