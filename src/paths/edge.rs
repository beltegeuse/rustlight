use crate::accel::*;
use crate::paths::path::*;
use crate::paths::vertex::*;
use crate::samplers::*;
use crate::scene::Scene;
use crate::structure::*;
use crate::volume::*;
use cgmath::*;

#[derive(Clone)]
pub struct Edge {
    /// Geometric informations
    pub dist: Option<f32>, // distance between points
    pub d: Vector3<f32>, // edge direction
    /// Connecting two vertices
    pub vertices: (VertexID, Option<VertexID>),
    /// Sampling information (from the BSDF or Phase function)
    pub sampled_distance: Option<SampledDistance>,
    pub pdf_direction: PDF,
    pub weight: Color, // BSDF * Transmittance
    pub contrib: Option<Color>,
    pub rr_weight: f32,
    pub id_sampling: usize,
}

impl Edge {
    pub fn from_vertex(
        path: &mut Path,
        org_vertex_id: VertexID,
        pdf_direction: PDF,
        weight: Color,
        contrib: Option<Color>,
        rr_weight: f32,
        next_vertex_id: VertexID,
        id_sampling: usize,
    ) -> EdgeID {
        let mut d = path.vertex(next_vertex_id).position() - path.vertex(org_vertex_id).position();
        let dist = d.magnitude();
        d /= dist;

        // Sampled distance is None here
        // as the distance inside the participating media
        // have not been sampled
        let edge = Edge {
            dist: Some(dist),
            d,
            vertices: (org_vertex_id, Some(next_vertex_id)),
            sampled_distance: None,
            pdf_direction,
            weight,
            contrib,
            rr_weight,
            id_sampling,
        };
        let edge = path.register_edge(edge);

        // This constructor have been only design for light vertex creation
        match path.vertex_mut(next_vertex_id) {
            Vertex::Light { edge_in, .. } => (*edge_in) = Some(edge),
            _ => unimplemented!(),
        };
        edge
    }

    pub fn from_ray<'scene>(
        path: &mut Path<'scene>,
        ray: &Ray,
        org_vertex_id: VertexID,
        pdf_direction: PDF,
        weight: Color,
        rr_weight: f32,
        sampler: &mut dyn Sampler,
        scene: &Scene,
        accel: &'scene dyn Acceleration,
        medium: Option<&HomogenousVolume>,
        id_sampling: usize,
    ) -> (EdgeID, Option<VertexID>) {
        let edge = Edge {
            dist: None,
            d: ray.d,
            vertices: (org_vertex_id, None),
            sampled_distance: None,
            pdf_direction,
            weight,
            contrib: None,
            rr_weight,
            id_sampling,
        };
        let edge = path.register_edge(edge);
        let its = match accel.trace(&ray) {
            Some(its) => its,
            None => {
                if let Some(ref m) = medium {
                    assert!(scene.emitter_environment.is_none());

                    // Sample the participating media
                    let mrec = m.sample(ray, sampler.next2d());
                    let pos = Point3::from_vec(ray.o.to_vec() + ray.d * mrec.t);
                    // We are sure to suceed as the distance is infine...
                    // TODO: Note that this design decision makes the env map incompatible with participating media presence
                    assert_eq!(mrec.exited, false);
                    let new_vertex = Vertex::Volume {
                        phase_function: PhaseFunction::Isotropic(),
                        pos,
                        d_in: -ray.d,
                        rr_weight: 1.0,
                        edge_in: edge,
                        edge_out: vec![],
                    };
                    let new_vertex = path.register_vertex(new_vertex);

                    // Update the edge
                    {
                        let edge = path.edge_mut(edge);
                        edge.dist = Some(mrec.t);
                        edge.vertices.1 = Some(new_vertex);
                        edge.weight *= mrec.w;
                        edge.sampled_distance = Some(mrec);
                    }

                    return (edge, Some(new_vertex));
                } else {
                    // Create an edge without distance
                    return (edge, None);
                }
            }
        };

        // Create the new vertex
        // This depends if there is a participating media or not
        let mut intersection_distance = its.dist;
        let (mrec, new_vertex) = if let Some(ref m) = medium {
            // Sample the participating media
            // Need to create a new ray as tfar need to store
            // the distance to the surface
            let mut ray_med = ray.clone();
            ray_med.tfar = intersection_distance;
            let mrec = m.sample(&ray_med, sampler.next2d());
            let new_vertex = if !mrec.exited {
                // Hit the volume
                // --- Update the distance
                intersection_distance = mrec.t;
                // --- Create the volume vertex
                let pos = Point3::from_vec(ray.o.to_vec() + ray.d * mrec.t);
                Vertex::Volume {
                    phase_function: PhaseFunction::Isotropic(),
                    pos,
                    d_in: -ray.d,
                    rr_weight: 1.0,
                    edge_in: edge,
                    edge_out: vec![],
                }
            } else {
                // Hit the surface
                Vertex::Surface {
                    its,
                    rr_weight: 1.0,
                    edge_in: edge,
                    edge_out: vec![],
                }
            };
            (Some(mrec), new_vertex)
        } else {
            (
                None,
                Vertex::Surface {
                    its,
                    rr_weight: 1.0,
                    edge_in: edge,
                    edge_out: vec![],
                },
            )
        };

        // Register the new vertex
        let new_vertex = path.register_vertex(new_vertex);

        // Update the edge information
        {
            let edge = path.edge_mut(edge);
            edge.dist = Some(intersection_distance);
            edge.vertices.1 = Some(new_vertex);
            if mrec.is_some() {
                edge.weight *= mrec.as_ref().unwrap().w;
            }
            edge.sampled_distance = mrec;
        }
        (edge, Some(new_vertex))
    }

    pub fn next_on_light_source(&self, scene: &Scene, path: &Path) -> bool {
        if let Some(v) = &self.vertices.1 {
            path.vertex(*v).on_light_source()
        } else {
            scene.emitter_environment.is_some()
        }
    }

    /// Get the contribution along this edge (toward the light direction)
    /// @deprecated: This might be not optimal as it is not a recursive call.
    pub fn contribution(&self, scene: &Scene, path: &Path) -> Color {
        if let Some(v) = &self.vertices.1 {
            match &self.contrib {
                Some(v) => *v * self.weight * self.rr_weight,
                None => self.weight * self.rr_weight * path.vertex(*v).contribution(self),
            }
        } else {
            self.weight * self.rr_weight * scene.enviroment_luminance(self.d)
        }
    }
}
