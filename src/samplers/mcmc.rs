use crate::samplers::*;
use cgmath::Point2;
use rand::prelude::*;

pub trait Mutator: Send {
    fn mutate(&self, v: f32, r: f32) -> f32;
    fn clone_box(&self) -> Box<dyn Mutator>;
}

#[derive(Clone)]
struct MutatorKelemen {
    pub s1: f32,
    pub s2: f32,
    log_ratio: f32,
}

impl MutatorKelemen {
    pub fn new(s1: f32, s2: f32) -> Self {
        MutatorKelemen {
            s1,
            s2,
            log_ratio: -(s2 / s1).ln(),
        }
    }
}

impl Default for MutatorKelemen {
    fn default() -> Self {
        MutatorKelemen::new(1.0 / 1024.0, 1.0 / 64.0)
    }
}

impl Mutator for MutatorKelemen {
    fn mutate(&self, v: f32, r: f32) -> f32 {
        let (add, r) = if r < 0.5 {
            (true, r * 2.0)
        } else {
            (false, 2.0 * (r - 0.5))
        };
        let dv = self.s2 * (r * self.log_ratio).exp();
        assert!(dv < 1.0);
        let mut v = if add {
            let mut v = v + dv;
            if v >= 1.0 {
                v -= 1.0
            }
            v
        } else {
            let mut v = v - dv;
            if v < 0.0 {
                v += 1.0
            }
            v
        };

        // TODO: This is a dirty fix for now.
        if v == 1.0 {
            v = 0.0;
        }
        assert!(v < 1.0);
        assert!(v >= 0.0);
        v
    }

    fn clone_box(&self) -> Box<dyn Mutator> {
        Box::new(self.clone())
    }
}

#[derive(Copy, Clone)]
pub struct SampleReplayValue {
    pub value: f32,
    pub modify: usize,
}

pub struct IndependentSamplerReplay {
    pub rnd: SmallRng,
    pub values: Vec<SampleReplayValue>,
    pub backup: Vec<(usize, SampleReplayValue)>,
    pub mutator: Box<dyn Mutator>,
    pub time: usize,
    pub time_large: usize,
    pub indice: usize,
    pub large_step: bool,
}

impl Sampler for IndependentSamplerReplay {
    fn next(&mut self) -> f32 {
        let v = self.sample(self.indice);
        self.indice += 1;
        v
    }

    fn next2d(&mut self) -> Point2<f32> {
        let v1 = self.sample(self.indice);
        let v2 = self.sample(self.indice + 1);
        self.indice += 2;
        Point2::new(v1, v2)
    }

    fn clone_box(&mut self) -> Box<dyn Sampler> {
        let rnd = rand::rngs::SmallRng::seed_from_u64(self.rnd.next_u64());
        Box::new(IndependentSamplerReplay {
            rnd,
            values: vec![],
            backup: vec![],
            mutator: self.mutator.clone_box(),
            time: 0,
            time_large: 0,
            indice: 0,
            large_step: false,
        })
    }

    // Nothing to do here!
    fn next_sample(&mut self) {}
    fn next_pixel(&mut self, _: Point2<u32>) {}
}

impl SamplerMCMC for IndependentSamplerReplay {
    fn accept(&mut self) {
        self.backup.clear();
        if self.large_step {
            self.time_large = self.time;
        }
        self.time += 1;
        self.indice = 0;
    }

    fn reject(&mut self) {
        if self.time == 0 {
            // This is just to catch error in case the random number
            // is wrongly initialize. With proper initialization via resampling
            // this case should never happens
            warn!("Reject state with time 0 (Maybe the chain was wrongly initialized)");
            self.values.clear();
        } else {
            for &(i, v) in &self.backup {
                self.values[i] = v;
            }
            self.backup.clear();
        }
        self.indice = 0;
    }
}

impl Default for IndependentSamplerReplay {
    fn default() -> Self {
        let rnd = rand::rngs::SmallRng::seed_from_u64(random());
        IndependentSamplerReplay {
            rnd,
            values: vec![],
            backup: vec![],
            mutator: Box::new(MutatorKelemen::default()),
            time: 0,
            time_large: 0,
            indice: 0,
            large_step: false,
        }
    }
}

//FIXME: Make not representable a sampler that are not accept
impl IndependentSamplerReplay {
    // Constructor to change the mutator technique
    pub fn mutator(mut self, mutator: Box<dyn Mutator>) -> Self {
        self.mutator = mutator;
        self
    }

    fn sample(&mut self, i: usize) -> f32 {
        while i >= self.values.len() {
            let value = self.rand();
            self.values.push(SampleReplayValue { value, modify: 0 })
        }

        if self.values[i].modify < self.time {
            if self.large_step {
                // In case of large step, we do a independent mutation
                self.backup.push((i, self.values[i]));
                self.values[i].value = self.rand();
                self.values[i].modify = self.time;
            } else {
                // Check if we need to do a large step
                if self.values[i].modify < self.time_large {
                    self.values[i].value = self.rand();
                    self.values[i].modify = self.time_large;
                }

                // Replay previous steps (up to time - 1)
                while self.values[i].modify + 1 < self.time {
                    let random = self.rand();
                    self.values[i].value = self.mutator.mutate(self.values[i].value, random);
                    self.values[i].modify += 1;
                }

                // The chain is now at time - 1 so we need to only
                // do one mutation
                self.backup.push((i, self.values[i]));
                let random = self.rand();
                self.values[i].value = self.mutator.mutate(self.values[i].value, random);
                self.values[i].modify += 1;
                assert_eq!(self.values[i].modify, self.time);
            }
        }

        self.values[i].value
    }

    // FIXME: Do not expose this function
    pub fn rand(&mut self) -> f32 {
        self.rnd.gen()
    }
}
