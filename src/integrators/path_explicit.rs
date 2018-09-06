use cgmath::Point2;
use integrators::*;
use paths::path::*;
use paths::vertex::*;
use scene::*;
use std::cell::RefCell;
use std::rc::Rc;
use structure::*;

pub struct IntegratorPathTracing {
    pub max_depth: Option<u32>,
}
pub struct TechniquePathTracing<'a, S: Sampler> {
    pub max_depth: Option<u32>,
    pub samplings: Vec<Box<SamplingStrategy<S>>>,
    pub root: Rc<RefCell<Vertex<'a>>>,
}
impl<'a, S: Sampler> Technique<'a, S> for TechniquePathTracing<'a, S> {
    fn evaluate(&self, scene: &'a Scene) -> Color {
        return self.evaluate_vertex(scene, &self.root);
    }

    fn init(&self, scene: &'a Scene, sampler: &mut S) -> Vec<(Rc<RefCell<Vertex<'a>>>, Color)> {
        // Only generate a path from the sensor
        return vec![(self.root.clone(), Color::one())];
    }

    fn expend(&self, _vertex: &Rc<RefCell<Vertex<'a>>>) -> bool {
        return true;
    }

    fn stratgies(&self, _vertex: &Rc<RefCell<Vertex<'a>>>) -> &Vec<Box<SamplingStrategy<S>>> {
        &self.samplings
    }
}
impl<'a, S: Sampler> TechniquePathTracing<'a, S> {
    fn evaluate_vertex(&self, scene: &'a Scene, vertex: &Rc<VertexPtr<'a>>) -> Color {
        let mut l_i = Color::zero();
        match *vertex.borrow() {
            Vertex::Surface(ref v) => {
                for edge in &v.edge_out {
                    let contrib = edge.borrow().contribution();
                    if !contrib.is_zero() {
                        let weight = if let PDF::SolidAngle(v) = edge.borrow().pdf_direction {
                            let total: f32 = self
                                .stratgies(vertex)
                                .iter()
                                .map(|s| {
                                    if let Some(v) = s.pdf(scene, &vertex, edge) {
                                        v
                                    } else {
                                        0.0
                                    }
                                })
                                .sum();
                            v / total
                        } else {
                            1.0
                        };
                        l_i += contrib * weight;
                    }

                    let edge = edge.borrow();
                    if let Some(ref vertex_next) = edge.vertices.1 {
                        l_i += edge.weight
                            * edge.rr_weight
                            * self.evaluate_vertex(scene, &vertex_next);
                    }
                }
            }
            Vertex::Sensor(ref v) => {
                // Only one strategy where...
                let edge = v.edge.as_ref().unwrap();

                // Get the potential contribution
                let contrib = edge.borrow().contribution();
                if !contrib.is_zero() {
                    l_i += contrib;
                }

                // Do the reccursive call
                if let Some(ref vertex_next) = edge.borrow().vertices.1 {
                    l_i += edge.borrow().weight * self.evaluate_vertex(scene, &vertex_next);
                }
            }
            _ => {}
        };
        return l_i;
    }
}

impl Integrator<Color> for IntegratorPathTracing {
    fn compute<'a, S: Sampler>(
        &self,
        (ix, iy): (u32, u32),
        scene: &'a Scene,
        sampler: &mut S,
    ) -> Color {
        self.compute_try((ix, iy), scene, sampler)
    }
}

impl IntegratorPathTracing {
    fn compute_try<'a, 'b: 'a, S: Sampler>(
        &self,
        (ix, iy): (u32, u32),
        scene: &'a Scene,
        sampler: &mut S,
    ) -> Color {
        // Initialize the technique
        let root = Rc::new(RefCell::new(Vertex::Sensor(SensorVertex {
            uv: Point2::new(ix as f32 + sampler.next(), iy as f32 + sampler.next()),
            pos: scene.camera.param.pos.clone(),
            edge: None,
        })));
        let mut samplings: Vec<Box<SamplingStrategy<S>>> = Vec::new();
        samplings.push(Box::new(DirectionalSamplingStrategy {}));
        samplings.push(Box::new(LightSamplingStrategy {}));
        let mut technique = TechniquePathTracing {
            max_depth: self.max_depth.clone(),
            samplings,
            root,
        };
        // Call the generator on this technique
        generate(scene, sampler, &mut technique);
        // And evaluate the sampling graph
        technique.evaluate(scene)
    }
}
