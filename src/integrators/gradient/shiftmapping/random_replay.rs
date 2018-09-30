use cgmath::Point2;
use integrators::gradient::shiftmapping::*;
use paths::path::*;
use paths::vertex::*;
use samplers::Sampler;
use scene::Scene;
use structure::Color;

// This special random number replay
// can capture the underlying sampler
// in order to replay the sequence of random number
// if it is necessary
pub struct ReplaySampler<'sampler, 'seq> {
    pub sampler: &'sampler mut Sampler,
    pub random: &'seq mut Vec<f32>,
    pub indice: usize,
}
impl<'sampler, 'seq> ReplaySampler<'sampler, 'seq> {
    fn generate(&mut self) -> f32 {
        assert!(self.indice <= self.random.len());
        if self.indice < self.random.len() {
            let v = self.indice;
            self.indice += 1;
            self.random[v]
        } else {
            let v = self.sampler.next();
            self.indice += 1;
            self.random.push(v);
            v
        }
    }
}
impl<'sampler, 'seq> Sampler for ReplaySampler<'sampler, 'seq> {
    fn next(&mut self) -> f32 {
        self.generate()
    }
    fn next2d(&mut self) -> Point2<f32> {
        let v1 = self.generate();
        let v2 = self.generate();
        Point2::new(v1, v2)
    }
}
pub struct RandomReplay {
    pub random_sequence: Vec<f32>,
    pub base_value: Color,
}
impl Default for RandomReplay {
    fn default() -> Self {
        RandomReplay {
            random_sequence: vec![],
            base_value: Color::zero(),
        }
    }
}
impl ShiftMapping for RandomReplay {
    fn base<'a>(
        &mut self,
        technique: &mut TechniqueGradientPathTracing,
        pos: Point2<u32>,
        scene: &'a Scene,
        sampler: &mut Sampler,
    ) -> (Color, Rc<VertexPtr<'a>>) {
        // Capture the random numbers
        let mut capture_sampler = ReplaySampler {
            sampler,
            random: &mut self.random_sequence,
            indice: 0,
        };
        // Call the generator on this technique
        // the generator give back the root nodes
        technique.img_pos = pos;
        let root = generate(scene, &mut capture_sampler, technique);
        let root = root[0].0.clone();
        self.base_value = technique.evaluate(scene, &root);
        (self.base_value, root)
    }
    fn shift<'a>(
        &mut self,
        technique: &mut TechniqueGradientPathTracing,
        pos: Point2<u32>,
        scene: &Scene,
        sampler: &mut Sampler,
        _base: &Rc<VertexPtr<'a>>,
    ) -> ShiftValue {
        technique.img_pos = pos;
        let mut capture_sampler = ReplaySampler {
            sampler,
            random: &mut self.random_sequence,
            indice: 0,
        };
        let offset = generate(scene, &mut capture_sampler, technique);
        let offset_contrib = technique.evaluate(scene, &offset[0].0);
        ShiftValue {
            base: 0.5 * self.base_value,
            offset: 0.5 * offset_contrib,
            gradient: 0.5 * (offset_contrib - self.base_value),
        }
    }
    fn clear(&mut self) {
        self.random_sequence.clear();
    }
}
