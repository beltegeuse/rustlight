use cgmath::Point2;
use integrators::*;
use paths::path::*;
use paths::vertex::*;
use std::cell::RefCell;
use std::rc::Rc;
use structure::*;

/// This structure store the rendering options
/// That the user have given through the command line
pub enum IntegratorPathTracingStrategies {
    All,
    BSDF,
    Emitter,
}
pub struct IntegratorPathTracing {
    pub max_depth: Option<u32>,
    pub strategy: IntegratorPathTracingStrategies,
}
/// This structure is responsible to the graph generation
pub struct TechniquePathTracing {
    pub max_depth: Option<u32>,
    pub samplings: Vec<Box<SamplingStrategy>>,
    pub img_pos: Point2<u32>,
}
impl<'a> Technique<'a> for TechniquePathTracing {
    fn init(
        &mut self,
        scene: &'a Scene,
        sampler: &mut Sampler,
    ) -> Vec<(Rc<RefCell<Vertex<'a>>>, Color)> {
        // Only generate a path from the sensor
        let root = Rc::new(RefCell::new(Vertex::Sensor(SensorVertex {
            uv: Point2::new(
                self.img_pos.x as f32 + sampler.next(),
                self.img_pos.y as f32 + sampler.next(),
            ),
            pos: scene.camera.position(),
            edge_in: None,
            edge_out: None,
        })));

        return vec![(root, Color::one())];
    }

    fn expand(&self, _vertex: &Rc<RefCell<Vertex<'a>>>, depth: u32) -> bool {
        self.max_depth.map_or(true, |max| depth < max)
    }

    fn strategies(&self, _vertex: &Rc<RefCell<Vertex<'a>>>) -> &Vec<Box<SamplingStrategy>> {
        &self.samplings
    }
}
impl TechniquePathTracing {
    fn evaluate<'a>(
        &self,
        scene: &'a Scene,
        vertex: &Rc<VertexPtr<'a>>,
        strategy: &IntegratorPathTracingStrategies,
    ) -> Color {
        let mut l_i = Color::zero();
        match *vertex.borrow() {
            Vertex::Surface(ref v) => {
                for edge in &v.edge_out {
                    let contrib = edge.borrow().contribution();
                    let contrib = match strategy {
                        IntegratorPathTracingStrategies::All => contrib,
                        IntegratorPathTracingStrategies::BSDF => {
                            if edge.borrow().id_sampling != 0 {
                                Color::zero()
                            } else {
                                contrib
                            }
                        }
                        IntegratorPathTracingStrategies::Emitter => {
                            if edge.borrow().id_sampling != 1 {
                                Color::zero()
                            } else {
                                contrib
                            }
                        }
                    };

                    if !contrib.is_zero() {
                        let weight = match strategy {
                            IntegratorPathTracingStrategies::All => {
                                // Balance heuristic
                                if let PDF::SolidAngle(v) = edge.borrow().pdf_direction {
                                    let total: f32 = self
                                        .strategies(vertex)
                                        .iter()
                                        .map(|s| {
                                            if let Some(v) = s.pdf(scene, &vertex, edge) {
                                                v
                                            } else {
                                                0.0
                                            }
                                        }).sum();
                                    v / total
                                } else {
                                    1.0
                                }
                            }
                            IntegratorPathTracingStrategies::BSDF
                            | IntegratorPathTracingStrategies::Emitter => 1.0,
                        };

                        l_i += contrib * weight;
                    }

                    let edge = edge.borrow();
                    if let Some(ref vertex_next) = edge.vertices.1 {
                        l_i += edge.weight
                            * edge.rr_weight
                            * self.evaluate(scene, &vertex_next, strategy);
                    }
                }
            }
            Vertex::Sensor(ref v) => {
                // Only one strategy where...
                let edge = v.edge_out.as_ref().unwrap();

                // Get the potential contribution
                let contrib = edge.borrow().contribution();
                if !contrib.is_zero() {
                    l_i += contrib;
                }

                // Do the reccursive call
                if let Some(ref vertex_next) = edge.borrow().vertices.1 {
                    l_i += edge.borrow().weight * self.evaluate(scene, &vertex_next, strategy);
                }
            }
            _ => {}
        };
        l_i
    }
}

impl Integrator for IntegratorPathTracing {
    fn compute(&mut self, scene: &Scene) -> Bitmap {
        compute_mc(self, scene)
    }
}
impl IntegratorMC for IntegratorPathTracing {
    fn compute_pixel(&self, (ix, iy): (u32, u32), scene: &Scene, sampler: &mut Sampler) -> Color {
        // Initialize the technique
        let mut samplings: Vec<Box<SamplingStrategy>> = Vec::new();
        samplings.push(Box::new(DirectionalSamplingStrategy {}));
        samplings.push(Box::new(LightSamplingStrategy {}));
        let mut technique = TechniquePathTracing {
            max_depth: self.max_depth,
            samplings,
            img_pos: Point2::new(ix, iy),
        };
        // Call the generator on this technique
        // the generator give back the root nodes
        let root = generate(scene, sampler, &mut technique);
        // Evaluate the sampling graph
        technique.evaluate(scene, &root[0].0, &self.strategy)
    }
}
