use integrators::*;
use paths::path::*;
use paths::vertex::*;
use scene::*;
use structure::*;

pub struct IntegratorUniPathNaive {
    pub max_depth: Option<u32>,
}

impl Integrator<Color> for IntegratorUniPathNaive {
    fn compute<S: Sampler>(&self, (ix, iy): (u32, u32), scene: &Scene, sampler: &mut S) -> Color {
        let mut samplings: Vec<Box<SamplingStrategy<S>>> = Vec::new();
        samplings.push(Box::new(DirectionalSamplingStrategy {})); 
        samplings.push(Box::new(LightSamplingStrategy {}));
        
        match Path::from_sensor((ix, iy), scene, sampler, self.max_depth, &samplings) {
            None => Color::zero(),
            Some(path) => path.evaluate(&samplings),
        }
    }
}