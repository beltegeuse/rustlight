use crate::integrators::*;
use crate::paths::path::*;
use crate::paths::strategies::*;
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
    pub rr_depth: Option<u32>,
    pub strategy: IntegratorPathTracingStrategies,
    pub single_scattering: bool,
}
/// This structure is responsible to the graph generation
pub struct TechniquePathTracing {
    pub max_depth: Option<u32>,
    pub samplings: Vec<Box<dyn SamplingStrategy>>,
    pub single_scattering: bool,
}
impl Technique for TechniquePathTracing {
    fn expand(&self, _vertex: &Vertex, depth: u32) -> bool {
        self.max_depth.map_or(true, |max| depth < max)
    }

    fn strategies(&self, _vertex: &Vertex) -> &Vec<Box<dyn SamplingStrategy>> {
        &self.samplings
    }
}
impl TechniquePathTracing {
    fn evalute_edge<'scene>(
        &self,
        curr_depth: u32,
        min_depth: Option<u32>,
        path: &Path<'scene>,
        scene: &'scene Scene,
        vertex_id: VertexID,
        edge_id: EdgeID,
        strategy: &IntegratorPathTracingStrategies,
    ) -> Color {
        // Get the edge that we considering
        let edge = path.edge(edge_id);
        // Compute the contribution
        let contrib = edge.contribution(scene, path);
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
                        let total = self
                            .strategies(path.vertex(vertex_id))
                            .iter()
                            .enumerate()
                            .map(|(id, s)| {
                                let pdf = if id == edge.id_sampling {
                                    v
                                } else {
                                    if let Some(v) = s.pdf(path, scene, vertex_id, edge_id) {
                                        v
                                    } else {
                                        0.0
                                    }
                                };
                                pdf
                            })
                            .sum::<f32>();
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

    fn evaluate<'scene>(
        &self,
        curr_depth: u32,
        min_depth: Option<u32>,
        path: &Path<'scene>,
        scene: &'scene Scene,
        vertex_id: VertexID,
        strategy: &IntegratorPathTracingStrategies,
    ) -> Color {
        if self.single_scattering && path.vertex(vertex_id).on_surface() {
            return Color::zero();
        }

        let mut l_i = Color::zero();
        match path.vertex(vertex_id) {
            Vertex::Surface { edge_out, .. } | Vertex::Volume { edge_out, .. } => {
                for edge_id in edge_out {
                    // Compute the contribution along this edge
                    // this only cover the fact that some next vertices are on some light sources
                    l_i += self.evalute_edge(
                        curr_depth, min_depth, path, scene, vertex_id, *edge_id, strategy,
                    );

                    // Continue on the edges if there is a vertex
                    let edge = path.edge(*edge_id);
                    if let Some(vertex_next_id) = edge.vertices.1 {
                        l_i += edge.weight
                            * edge.rr_weight
                            * self.evaluate(
                                curr_depth + 1,
                                min_depth,
                                path,
                                scene,
                                vertex_next_id,
                                strategy,
                            );
                    }
                }
            }
            Vertex::Sensor { edge_out, .. } => {
                // Only one strategy where...
                let edge = path.edge(edge_out.unwrap());

                let add_contrib = match min_depth {
                    None => true,
                    Some(v) => curr_depth >= v,
                };

                // Get the potential contribution
                let contrib = edge.contribution(scene, path);
                if !contrib.is_zero() && add_contrib {
                    l_i += contrib;
                }

                // Do the reccursive call
                if let Some(vertex_next_id) = edge.vertices.1 {
                    l_i += edge.weight
                        * edge.rr_weight
                        * self.evaluate(
                            curr_depth + 1,
                            min_depth,
                            path,
                            scene,
                            vertex_next_id,
                            strategy,
                        );
                }
            }
            _ => {}
        };
        l_i
    }
}

impl Integrator for IntegratorPathTracing {
    fn compute(
        &mut self,
        sampler: &mut dyn Sampler,
        accel: &dyn Acceleration,
        scene: &Scene,
    ) -> BufferCollection {
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
    ) -> Color {
        // Initialize the technique
        let mut samplings: Vec<Box<dyn SamplingStrategy>> = Vec::new();

        // Always need the directional strategy to expend the path
        samplings.push(Box::new(
            crate::paths::strategies::directional::DirectionalSamplingStrategy {
                transport: Transport::Importance,
                rr_depth: self.rr_depth,
            },
        ));
        match self.strategy {
            IntegratorPathTracingStrategies::All | IntegratorPathTracingStrategies::Emitter => {
                // This strategy only make sense in case of light sampling
                samplings.push(Box::new(
                    crate::paths::strategies::emitters::LightSamplingStrategy {},
                ));
            }
            _ => {}
        }
        // Create the technique responsible for the actual tracing
        let mut technique = TechniquePathTracing {
            max_depth: self.max_depth,
            samplings,
            single_scattering: self.single_scattering,
        };
        // Call the generator on this technique
        // the generator give back the root nodes
        let mut path = Path::default();
        let root = path.from_sensor(Point2::new(ix, iy), scene, sampler);
        generate(&mut path, root.0, accel, scene, sampler, &mut technique);
        // Evaluate the sampling graph
        technique.evaluate(0, self.min_depth, &path, scene, root.0, &self.strategy)
    }
}
