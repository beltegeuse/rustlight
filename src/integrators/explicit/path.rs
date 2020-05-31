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
    pub min_depth: Option<u32>,
    pub max_depth: Option<u32>,
    pub strategy: IntegratorPathTracingStrategies,
    pub single_scattering: bool,
}
/// This structure is responsible to the graph generation
pub struct TechniquePathTracing {
    pub max_depth: Option<u32>,
    pub samplings: Vec<Box<dyn SamplingStrategy>>,
    pub img_pos: Point2<u32>,
    pub single_scattering: bool,
}
impl Technique for TechniquePathTracing {
    fn init<'scene, 'emitter>(
        &mut self,
        path: &mut Path<'scene, 'emitter>,
        _accel: &dyn Acceleration,
        scene: &'scene Scene,
        sampler: &mut dyn Sampler,
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

    fn strategies(&self, _vertex: &Vertex) -> &Vec<Box<dyn SamplingStrategy>> {
        &self.samplings
    }
}
impl TechniquePathTracing {
    fn evalute_edge<'scene, 'emitter>(
        &self,
        curr_depth: u32,
        min_depth: Option<u32>,
        path: &Path<'scene, 'emitter>,
        scene: &'scene Scene,
        emitters: &'emitter EmitterSampler,
        vertex_id: VertexID,
        edge_id: EdgeID,
        strategy: &IntegratorPathTracingStrategies,
    ) -> Color {
        // Get the edge that we considering
        let edge = path.edge(edge_id);
        // Compute the contribution
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

        // Check the path depth, if not reach min depth, ignore contribution...
        let add_contrib = match min_depth {
            None => true,
            Some(v) => curr_depth >= v, 
        };

        // If needed, we compute the MIS weight
        if !contrib.is_zero() && add_contrib {
            let weight = match strategy {
                IntegratorPathTracingStrategies::All => {
                    // Balance heuristic
                    if let PDF::SolidAngle(v) = edge.pdf_direction {
                        let total: f32 = self
                            .strategies(path.vertex(vertex_id))
                            .iter()
                            .map(|s| {
                                if let Some(v) = s.pdf(path, scene, emitters, vertex_id, edge_id) {
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
                // No MIS in this case
                IntegratorPathTracingStrategies::BSDF
                | IntegratorPathTracingStrategies::Emitter => 1.0,
            };
            contrib * weight
        } else {
            Color::zero()
        }
    }

    fn evaluate<'scene, 'emitter>(
        &self,
        curr_depth: u32,
        min_depth: Option<u32>,
        path: &Path<'scene, 'emitter>,
        scene: &'scene Scene,
        emitters: &'emitter EmitterSampler,
        vertex_id: VertexID,
        strategy: &IntegratorPathTracingStrategies,
    ) -> Color {
        let mut l_i = Color::zero();
        match path.vertex(vertex_id) {
            Vertex::Surface(ref v) => {
                if self.single_scattering {
                    return Color::zero();
                }
                for edge_id in &v.edge_out {
                    // Compute the contribution along this edge
                    // this only cover the fact that some next vertices are on some light sources
                    // TODO: Modify this scheme at some point
                    l_i += self.evalute_edge(curr_depth, min_depth, path, scene, emitters, vertex_id, *edge_id, strategy);

                    // Continue on the edges if there is a vertex
                    let edge = path.edge(*edge_id);
                    if let Some(vertex_next_id) = edge.vertices.1 {
                        l_i += edge.weight
                            * edge.rr_weight
                            * self.evaluate(curr_depth + 1, min_depth,path, scene, emitters, vertex_next_id, strategy);
                    }
                }
            }
            Vertex::Volume(ref v) => {
                for edge_id in &v.edge_out {
                    // Compute the contribution along this edge
                    // this only cover the fact that some next vertices are on some light sources
                    // TODO: Modify this scheme at some point
                    l_i += self.evalute_edge(curr_depth, min_depth, path, scene, emitters, vertex_id, *edge_id, strategy);

                    // Continue on the edges if there is a vertex
                    let edge = path.edge(*edge_id);

                    if let Some(vertex_next_id) = edge.vertices.1 {
                        l_i += edge.weight
                            * edge.rr_weight
                            * self.evaluate(curr_depth + 1, min_depth, path, scene, emitters, vertex_next_id, strategy);
                    }
                }
            }
            Vertex::Sensor(ref v) => {
                // Only one strategy where...
                let edge = path.edge(v.edge_out.unwrap());

                let add_contrib = match min_depth {
                    None => true,
                    Some(v) => curr_depth >= v, 
                };

                // Get the potential contribution
                let contrib = edge.contribution(path);
                if !contrib.is_zero() && add_contrib {
                    l_i += contrib;
                }

                // Do the reccursive call
                if let Some(vertex_next_id) = edge.vertices.1 {
                    l_i += edge.weight
                        * edge.rr_weight
                        * self.evaluate(curr_depth + 1, min_depth, path, scene, emitters, vertex_next_id, strategy);
                }
            }
            _ => {}
        };
        l_i
    }
}

impl Integrator for IntegratorPathTracing {
    fn compute(&mut self, sampler: &mut dyn Sampler, accel: &dyn Acceleration, scene: &Scene) -> BufferCollection {
        compute_mc(self, sampler, accel, scene)
    }
}
impl IntegratorMC for IntegratorPathTracing {
    fn compute_pixel(
        &self,
        (ix, iy): (u32, u32),
        accel: &dyn Acceleration,
        scene: &Scene,
        sampler: &mut dyn Sampler,
        emitters: &EmitterSampler,
    ) -> Color {
        // Initialize the technique
        let mut samplings: Vec<Box<dyn SamplingStrategy>> = Vec::new();

        // Always need the directional strategy to expend the path
        samplings.push(Box::new(DirectionalSamplingStrategy { from_sensor: true }));
        match self.strategy {
            IntegratorPathTracingStrategies::All | IntegratorPathTracingStrategies::Emitter => {
                // This strategy only make sense in case of light sampling
                samplings.push(Box::new(LightSamplingStrategy {}));
            }
            _ => {}
        }

        // Create the technique responsible for the actual tracing
        let mut technique = TechniquePathTracing {
            max_depth: self.max_depth,
            samplings,
            img_pos: Point2::new(ix, iy),
            single_scattering: self.single_scattering,
        };
        // Call the generator on this technique
        // the generator give back the root nodes
        let mut path = Path::default();
        let root = generate(&mut path, accel, scene, emitters, sampler, &mut technique);
        // Evaluate the sampling graph
        technique.evaluate(0, self.min_depth, &path, scene, emitters, root[0].0, &self.strategy)
    }
}
