use cgmath::Point2;

pub trait Sampler: Send {
    fn next(&mut self) -> f32;
    fn next_u64(&mut self) -> u64 {
        unimplemented!("Not implemented");
    }
    fn next2d(&mut self) -> Point2<f32>;
    fn clone(&mut self) -> Box<dyn Sampler> {
        unimplemented!("Clone not implemented");
    }
}

pub trait SamplerMCMC {
    fn accept(&mut self);
    fn reject(&mut self);
}

pub mod independent;
pub mod mcmc;
