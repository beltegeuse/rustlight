use crate::accel::*;
use crate::emitter::*;
use crate::paths::edge::*;
use crate::paths::path::*;
use crate::paths::strategy::*;
use crate::paths::vertex::*;
use crate::samplers::*;
use crate::scene::*;
use crate::structure::*;
use crate::volume::*;
use cgmath::InnerSpace;

pub struct LightSamplingStrategy {}
impl LightSamplingStrategy {
    fn pdf_emitter<'scene, 'emitter>(
        &self,
        path: &Path<'scene, 'emitter>,
        emitters: &'emitter EmitterSampler,
        ray: Ray,
        next_vertex_id: VertexID,
    ) -> Option<f32> {
        match path.vertex(next_vertex_id) {
            Vertex::Surface { its, .. } => {
                // We could create a emitter sampling
                // if we have intersected the light source randomly
                if let PDF::SolidAngle(light_pdf) =
                    emitters.direct_pdf(its.mesh, &LightSamplingPDF::new(&ray, its))
                {
                    Some(light_pdf)
                } else {
                    None
                }
            }
            Vertex::Light {
                emitter, pos, n, ..
            } => {
                if let PDF::SolidAngle(light_pdf) = emitters.direct_pdf(
                    *emitter,
                    &LightSamplingPDF {
                        o: ray.o,
                        p: *pos,
                        n: *n,
                        dir: ray.d,
                    },
                ) {
                    Some(light_pdf)
                } else {
                    None
                }
            }
            Vertex::Sensor { .. } | Vertex::Volume { .. } => None,
        }
    }
}
impl SamplingStrategy for LightSamplingStrategy {
    fn sample<'scene, 'emitter>(
        &self,
        path: &mut Path<'scene, 'emitter>,
        vertex_id: VertexID,
        accel: &'scene dyn Acceleration,
        _scene: &'scene Scene,
        emitters: &'emitter EmitterSampler,
        _throughput: Color,
        sampler: &mut dyn Sampler,
        medium: Option<&HomogenousVolume>,
        id_strategy: usize,
    ) -> Option<(VertexID, Color)> {
        let (edge, _next_vertex) = match path.vertex(vertex_id) {
            Vertex::Surface { its, .. } => {
                if its.mesh.bsdf.is_smooth() {
                    return None;
                }

                // Generate the light sampling record based on the current vertex location
                // Note that during this procedure, we did not evaluate the product of the path throughput
                // and the incomming direct light. This evaluation will be done later when MIS
                // will be computed.
                let light_record =
                    emitters.sample_light(&its.p, sampler.next(), sampler.next(), sampler.next2d());
                let visible = accel.visible(&its.p, &light_record.p);
                if light_record.is_valid() && visible {
                    // We create a new vertex as it is a light
                    let next_vertex = Vertex::Light {
                        pos: light_record.p,
                        n: light_record.n,
                        emitter: light_record.emitter,
                        edge_in: None,
                        edge_out: None,
                    };

                    // FIXME: Only work for diffuse light
                    // FIXME: Check the direction of hte light
                    let mut weight = light_record.weight;
                    let emission = light_record.emitter.emitted_luminance(light_record.d);
                    weight.r /= emission.r;
                    weight.g /= emission.g;
                    weight.b /= emission.b;

                    // Need to evaluate the BSDF
                    weight *= &its.mesh.bsdf.eval(
                        &its.uv,
                        &its.wi,
                        &its.to_local(&light_record.d),
                        Domain::SolidAngle,
                        Transport::Importance,
                    );

                    if let Some(m) = medium {
                        // Evaluate the transmittance
                        let mut ray = Ray::new(its.p, light_record.d);
                        let d = light_record.p - its.p;
                        // Trick to compute the distance
                        ray.tfar = d.dot(light_record.d);
                        assert!(ray.tfar > 0.0);
                        // Compute the transmittance
                        let transmittance = m.transmittance(ray);
                        weight *= transmittance;
                    }

                    let next_vertex_id = path.register_vertex(next_vertex);
                    (
                        Edge::from_vertex(
                            path,
                            vertex_id,
                            light_record.pdf,
                            weight,
                            1.0,
                            next_vertex_id,
                            id_strategy,
                        ),
                        next_vertex_id,
                    )
                } else {
                    return None;
                }
            }
            Vertex::Volume {
                pos,
                phase_function,
                d_in,
                ..
            } => {
                // Generate the light sampling record based on the current vertex location
                // Note that during this procedure, we did not evaluate the product of the path throughput
                // and the incomming direct light. This evaluation will be done later when MIS
                // will be computed.
                let light_record =
                    emitters.sample_light(&pos, sampler.next(), sampler.next(), sampler.next2d());
                let visible = accel.visible(&pos, &light_record.p);
                if light_record.is_valid() && visible {
                    let next_vertex = Vertex::Light {
                        pos: light_record.p,
                        n: light_record.n,
                        emitter: light_record.emitter,
                        edge_in: None,
                        edge_out: None,
                    };

                    // FIXME: Only work for diffuse light
                    // FIXME: This is the wrong -d_out_local, no?
                    let mut weight = light_record.weight;
                    let emission = light_record.emitter.emitted_luminance(-light_record.d);
                    weight.r /= emission.r;
                    weight.g /= emission.g;
                    weight.b /= emission.b;

                    // Need to evaluate the phase function
                    weight *= &phase_function.eval(d_in, &light_record.d);

                    if let Some(m) = medium {
                        // Evaluate the transmittance
                        let mut ray = Ray::new(*pos, light_record.d);
                        let d = light_record.p - pos;
                        // Trick to compute the distance
                        ray.tfar = d.dot(light_record.d);
                        assert!(ray.tfar > 0.0);
                        // Generate the ray and compute the transmittance
                        let transmittance = m.transmittance(ray);
                        weight *= transmittance;
                    }

                    let next_vertex_id = path.register_vertex(next_vertex);
                    (
                        Edge::from_vertex(
                            path,
                            vertex_id,
                            light_record.pdf,
                            weight,
                            1.0,
                            next_vertex_id,
                            id_strategy,
                        ),
                        next_vertex_id,
                    )
                } else {
                    return None;
                }
            }
            _ => return None,
        };

        // Update the out edge
        match path.vertex_mut(vertex_id) {
            Vertex::Surface { edge_out, .. } | Vertex::Volume { edge_out, .. } => {
                edge_out.push(edge);
            }
            _ => unimplemented!(),
        }

        None // Finish the sampling here
    }

    fn pdf<'scene, 'emitter>(
        &self,
        path: &Path<'scene, 'emitter>,
        _scene: &'scene Scene,
        emitters: &'emitter EmitterSampler,
        vertex_id: VertexID,
        edge_id: EdgeID,
    ) -> Option<f32> {
        // Get the edge and check the condition
        let edge = path.edge(edge_id);
        if !edge.next_on_light_source(path) {
            return None;
        }

        // Retrive the proper pdf in this case
        let vertex = path.vertex(vertex_id);
        match vertex {
            Vertex::Volume { .. } => {
                // Always ok for have sampling a light source
                let ray = Ray::new(vertex.position(), edge.d);
                if let Some(next_vertex_id) = edge.vertices.1 {
                    self.pdf_emitter(path, emitters, ray, next_vertex_id)
                } else {
                    None
                }
            }
            Vertex::Surface { its, .. } => {
                // Impossible to sample from a Dirac distribution
                if its.mesh.bsdf.is_smooth() {
                    return None;
                }
                // Know the the light is intersectable so have a solid angle PDF
                let ray = Ray::new(vertex.position(), edge.d);
                if let Some(next_vertex_id) = edge.vertices.1 {
                    self.pdf_emitter(path, emitters, ray, next_vertex_id)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}
