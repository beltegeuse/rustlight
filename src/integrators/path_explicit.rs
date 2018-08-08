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
        match Path::from_sensor((ix, iy), scene, sampler, self.max_depth) {
            None => Color::zero(),
            Some(path) => path.evaluate(),
        }
    }
}

pub struct IntegratorUniPath {
    pub max_depth: Option<u32>,
}
impl Integrator<Color> for IntegratorUniPath {
    fn compute<S: Sampler>(&self, (ix, iy): (u32, u32), scene: &Scene, sampler: &mut S) -> Color {
        match Path::from_sensor((ix, iy), scene, sampler, self.max_depth) {
            None => Color::zero(),
            Some(path) => {
                let path = PathWithDirect::generate(scene, sampler, path, self.max_depth);
                path.evaluate(scene)
            }
        }
    }
}
