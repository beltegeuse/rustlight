use crate::integrators::*;
use crate::paths::path::*;
use crate::paths::vertex::*;
use cgmath::Point2;

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
impl Technique for TechniquePathTracing {
    fn init<'scene, 'emitter>(
        &mut self,
        path: &mut Path<'scene, 'emitter>,
        _accel: &Acceleration,
        scene: &'scene Scene,
        sampler: &mut Sampler,
        _emitters: &'emitter EmitterSampler,
    ) -> Vec<(VertexID, Color)> {
        // Only generate a path from the sensor
        let root = Vertex::Sensor(SensorVertex {
            uv: Point2::new(
                self.img_pos.x as f32 + sampler.next(),
                self.img_pos.y as f32 + sampler.next(),
            ),
            pos: scene.camera.position(),
            edge_in: None,
            edge_out: None,
        });

        return vec![(path.register_vertex(root), Color::one())];
    }

    fn expand(&self, _vertex: &Vertex, depth: u32) -> bool {
        self.max_depth.map_or(true, |max| depth < max)
    }

    fn strategies(&self, _vertex: &Vertex) -> &Vec<Box<SamplingStrategy>> {
        &self.samplings
    }
}
impl TechniquePathTracing {
    fn evaluate<'scene, 'emitter>(
        &self,
        path: &Path<'scene, 'emitter>,
        scene: &'scene Scene,
        emitters: &'emitter EmitterSampler,
        vertex_id: VertexID,
        strategy: &IntegratorPathTracingStrategies,
    ) -> Color {
        let mut l_i = Color::zero();
        match path.vertex(vertex_id) {
            Vertex::Surface(ref v) => {
                for edge_id in &v.edge_out {
                    let edge = path.edge(*edge_id);
                    let contrib = edge.contribution(path);
                    let contrib = match strategy {
                        IntegratorPathTracingStrategies::All => contrib,
                        IntegratorPathTracingStrategies::BSDF => {
                            if edge.id_sampling != 0 {
                                Color::zero()
                            } else {
                                contrib
                            }
                        }
                        IntegratorPathTracingStrategies::Emitter => {
                            if edge.id_sampling != 1 {
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
                                if let PDF::SolidAngle(v) = edge.pdf_direction {
                                    let total: f32 = self.strategies(path.vertex(vertex_id))
                                        .iter()
                                        .map(|s| {
                                            if let Some(v) =
                                                s.pdf(path, scene, emitters, vertex_id, *edge_id)
                                            {
                                                v
                                            } else {
                                                0.0
                                            }
                                        })
                                        .sum();
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

                    if let Some(vertex_next_id) = edge.vertices.1 {
                        l_i += edge.weight * edge.rr_weight
                            * self.evaluate(path, scene, emitters, vertex_next_id, strategy);
                    }
                }
            }
            Vertex::Sensor(ref v) => {
                // Only one strategy where...
                let edge = path.edge(v.edge_out.unwrap());

                // Get the potential contribution
                let contrib = edge.contribution(path);
                if !contrib.is_zero() {
                    l_i += contrib;
                }

                // Do the reccursive call
                if let Some(vertex_next_id) = edge.vertices.1 {
                    l_i += edge.weight
                        * self.evaluate(path, scene, emitters, vertex_next_id, strategy);
                }
            }
            _ => {}
        };
        l_i
    }
}

impl Integrator for IntegratorPathTracing {
    fn compute(&mut self, accel: &Acceleration, scene: &Scene) -> BufferCollection {
        compute_mc(self, accel, scene)
    }
}
impl IntegratorMC for IntegratorPathTracing {
    fn compute_pixel(
        &self,
        (ix, iy): (u32, u32),
        accel: &Acceleration,
        scene: &Scene,
        sampler: &mut Sampler,
        emitters: &EmitterSampler,
    ) -> Color {
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
        let mut path = Path::default();
        let root = generate(&mut path, accel, scene, emitters, sampler, &mut technique);
        // Evaluate the sampling graph
        technique.evaluate(&path, scene, emitters, root[0].0, &self.strategy)
    }
}
