use cgmath::Point2;

pub trait Sampler: Send {
    fn next(&mut self) -> f32;
    fn next2d(&mut self) -> Point2<f32>;
}

pub trait SamplerMCMC {
    fn accept(&mut self);
    fn reject(&mut self);
}

pub mod independent;
pub mod mcmc;
