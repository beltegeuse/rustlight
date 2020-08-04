use cgmath::Point2;

pub trait Sampler: Send {
    fn next(&mut self) -> f32;
    fn next2d(&mut self) -> Point2<f32>;
    fn clone_box(&mut self) -> Box<dyn Sampler>;
    fn next_sample(&mut self);
    fn next_pixel(&mut self, p: Point2<u32>);
}

pub trait SamplerMCMC {
    fn accept(&mut self);
    fn reject(&mut self);
}

pub mod independent;
pub mod mcmc;
pub mod stratified;
