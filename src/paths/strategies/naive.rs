use crate::accel::*;
use crate::math::*;
use crate::paths::edge::*;
use crate::paths::path::*;
use crate::paths::strategies::*;
use crate::Scale;
use cgmath::InnerSpace;

pub struct NaiveSamplingStrategy {
    pub transport: Transport,
    pub rr_depth: Option<u32>,
}
impl NaiveSamplingStrategy {
    pub fn bounce<'scene>(
        &self,
        path: &mut Path<'scene>,
        vertex_id: VertexID,
        accel: &'scene dyn Acceleration,
        scene: &'scene Scene,
        throughput: &mut Color,
        sampler: &mut dyn Sampler,
        medium: Option<&HomogenousVolume>,
        id_strategy: usize,
        depth: u32,
    ) -> (Option<EdgeID>, Option<VertexID>) {
        match path.vertex(vertex_id) {
            Vertex::Sensor { uv, .. } => {
                // Generate the path from the sensor
                let ray = scene.camera.generate(*uv);
                let (edge, new_vertex) = Edge::from_ray(
                    path,
                    &ray,
                    vertex_id,
                    PDF::SolidAngle(1.0),
                    Color::one(),
                    1.0,
                    sampler,
                    scene,
                    accel,
                    medium,
                    id_strategy,
                );
                (Some(edge), new_vertex)
            }
            Vertex::Surface { its, .. } => {
                // TODO: Should depends of the domain...
                let d = crate::math::cosine_sample_hemisphere(sampler.next2d());
                let d_out_global = its.frame.to_world(d);
                let pdf = d.z.abs() / std::f32::consts::PI;
                // TODO: This is fine with debug, but might be too costly for production
                // Make sure that we get a valid outgoing direction
                assert_approx_eq!(d_out_global.dot(d_out_global), 1.0, 0.0001);

                // Update the throughput
                let (domain, pdf) = if its.mesh.bsdf.bsdf_type().is_smooth() {
                    unimplemented!();
                // (Domain::Discrete, PDF::Discrete(pdf))
                } else {
                    (Domain::SolidAngle, PDF::SolidAngle(pdf))
                };
                let bsdf_weight = its
                    .mesh
                    .bsdf
                    .eval(&its.uv, &its.wi, &d, domain, self.transport)
                    / pdf.value();
                *throughput *= &bsdf_weight;

                // TODO: Need to further test this part
                // TODO: This might be problematic for BDPT implementation
                if self.transport == Transport::Radiance {
                    let wi_global = its.frame.to_world(its.wi);
                    let correction =
                        (its.wi.z * d_out_global.dot(its.n_g)) / (d.z * wi_global.dot(its.n_g));
                    *throughput *= correction.abs();
                }

                if throughput.is_zero() {
                    return (None, None);
                }

                // Check RR
                let do_rr = match self.rr_depth {
                    None => true,
                    Some(v) => v <= depth,
                };
                let rr_weight = if do_rr {
                    let rr_weight = throughput.channel_max().min(0.95);
                    if rr_weight < sampler.next() {
                        return (None, None);
                    }
                    1.0 / rr_weight
                } else {
                    1.0
                };
                throughput.scale(rr_weight);

                // Generate the new ray and do the intersection
                let ray = Ray::new(its.p, d_out_global);
                let (edge, new_vertex) = Edge::from_ray(
                    path,
                    &ray,
                    vertex_id,
                    pdf,
                    bsdf_weight,
                    rr_weight,
                    sampler,
                    scene,
                    accel,
                    medium,
                    id_strategy,
                );
                return (Some(edge), new_vertex);
            }
            Vertex::Volume {
                phase_function,
                d_in,
                pos,
                ..
            } => {
                let d = sample_uniform_sphere(sampler.next2d());
                let pdf = PDF::SolidAngle(1.0 / (std::f32::consts::PI * 2.0));
                let weight = phase_function.eval(&d_in, &d) / pdf.value();

                // Update the throughput
                *throughput *= &weight;
                if throughput.is_zero() {
                    return (None, None);
                }

                // Check RR
                let do_rr = match self.rr_depth {
                    None => true,
                    Some(v) => v <= depth,
                };
                let rr_weight = if do_rr {
                    let rr_weight = throughput.channel_max().min(0.95);
                    if rr_weight < sampler.next() {
                        return (None, None);
                    }
                    1.0 / rr_weight
                } else {
                    1.0
                };
                throughput.scale(rr_weight);

                // Generate the new ray and do the intersection
                let ray = Ray::new(*pos, d);
                let (edge, new_vertex) = Edge::from_ray(
                    path,
                    &ray,
                    vertex_id,
                    pdf,
                    weight,
                    rr_weight,
                    sampler,
                    scene,
                    accel,
                    medium,
                    id_strategy,
                );
                (Some(edge), new_vertex)
            }
            Vertex::Light { n, pos, .. } => {
                // For now, just computing the outgoing direction
                // Using cosine base weighting as we know that the light source
                // can only be cosine based isotropic lighting
                let d_out = cosine_sample_hemisphere(sampler.next2d());
                if d_out.z == 0.0 {
                    return (None, None); // Failed to sample the outgoing direction
                }

                let frame = Frame::new(*n);
                let d_out_global = frame.to_world(d_out);
                let ray = Ray::new(*pos, d_out_global);
                // FIXME: This might be wrong!
                let weight = Color::one(); // Perfectly importance sampled

                // This will generate the edge
                // if there is a participating media
                // the edge will be generated properly
                let (edge, new_vertex) = Edge::from_ray(
                    path,
                    &ray,
                    vertex_id,
                    PDF::SolidAngle(d_out.z * std::f32::consts::FRAC_1_PI),
                    weight,
                    1.0,
                    sampler,
                    scene,
                    accel,
                    medium,
                    id_strategy,
                );

                (Some(edge), new_vertex)
            }
        }
    }
}
impl SamplingStrategy for NaiveSamplingStrategy {
    fn sample<'scene>(
        &self,
        path: &mut Path<'scene>,
        vertex_id: VertexID,
        accel: &'scene dyn Acceleration,
        scene: &'scene Scene,
        mut throughput: Color,
        sampler: &mut dyn Sampler,
        medium: Option<&HomogenousVolume>,
        id_strategy: usize,
        depth: u32,
    ) -> Option<(VertexID, Color)> {
        // Generate the next edge and the next vertex
        let (edge, new_vertex) = self.bounce(
            path,
            vertex_id,
            accel,
            scene,
            &mut throughput,
            sampler,
            medium,
            id_strategy,
            depth,
        );

        // Update the edge if we sucesfull sample it
        if let Some(e) = edge {
            match path.vertex_mut(vertex_id) {
                Vertex::Sensor { edge_out, .. } | Vertex::Light { edge_out, .. } => {
                    // For light tracing
                    // note that the direction of light
                    // if not correct in this case
                    (*edge_out) = Some(e);
                }
                Vertex::Surface { edge_out, .. } | Vertex::Volume { edge_out, .. } => {
                    edge_out.push(e);
                }
            }
        }

        if let Some(new_vertex) = new_vertex {
            Some((new_vertex, throughput))
        } else {
            None
        }
    }
    fn pdf<'scene>(
        &self,
        path: &Path<'scene>,
        scene: &'scene Scene,
        vertex_id: VertexID,
        edge_id: EdgeID,
    ) -> Option<f32> {
        // FIXME: It seems to have a design flaw here
        // here the pdf here is the light sampling opponent
        // in this case, it makes sense that the PDF for this strategy
        // if None in case of delta distribution...
        let edge = path.edge(edge_id);
        if !edge.next_on_light_source(scene, path) {
            return None;
        }

        // FIXME: Add the PDF of sampling between the two points
        // TODO: Do we need to store this information somehow here?
        //       as it might be required in case of heterogenous PM

        match path.vertex(vertex_id) {
            Vertex::Surface { its, .. } => {
                // TODO: Check why in the case of smooth, we cannot sample the light source...
                if its.mesh.bsdf.bsdf_type().is_smooth() {
                    return None;
                }
                if let PDF::SolidAngle(pdf) = its.mesh.bsdf.pdf(
                    &its.uv,
                    &its.wi,
                    &its.frame.to_local(edge.d),
                    Domain::SolidAngle,
                    self.transport,
                ) {
                    return Some(pdf);
                }
                unimplemented!();
            }
            Vertex::Volume {
                phase_function,
                d_in,
                ..
            } => Some(phase_function.pdf(d_in, &edge.d)),
            Vertex::Sensor { .. } => Some(1.0), // TODO: Why this value?
            Vertex::Light { .. } => None,       // Impossible to do BSDF sampling on a light source
        }
    }
}
