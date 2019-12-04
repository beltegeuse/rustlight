use crate::emitter::*;
use crate::math::*;
use crate::paths::vertex::*;
use crate::samplers::*;
use crate::scene::*;
use crate::structure::*;
use crate::Scale;
use std;
use std::mem;

pub trait SamplingStrategy {
    fn sample<'scene, 'emitter>(
        &self,
        path: &mut Path<'scene, 'emitter>,
        vertex_id: VertexID,
        accel: &'scene Acceleration,
        scene: &'scene Scene,
        emitters: &'emitter EmitterSampler,
        throughput: Color,
        sampler: &mut Sampler,
        id_strategy: usize,
    ) -> Option<(VertexID, Color)>;

    // All PDF have to be inside the same domain
    fn pdf<'scene, 'emitter>(
        &self,
        path: &Path<'scene, 'emitter>,
        scene: &'scene Scene,
        emitters: &'emitter EmitterSampler,
        vertex_id: VertexID,
        edge_id: EdgeID,
    ) -> Option<f32>;
}

pub struct DirectionalSamplingStrategy {}
impl DirectionalSamplingStrategy {
    pub fn bounce<'scene>(
        path: &mut Path<'scene, '_>,
        vertex_id: VertexID,
        accel: &'scene Acceleration,
        scene: &'scene Scene,
        throughput: &mut Color,
        sampler: &mut Sampler,
        id_strategy: usize,
    ) -> (Option<EdgeID>, Option<VertexID>) {
        match path.vertex(vertex_id) {
            Vertex::Sensor(ref v) => {
                let ray = scene.camera.generate(v.uv);
                let (edge, new_vertex) = Edge::from_ray(
                    path,
                    &ray,
                    vertex_id,
                    PDF::SolidAngle(1.0),
                    Color::one(),
                    1.0,
                    accel,
                    scene,
                    id_strategy,
                );
                (Some(edge), new_vertex)
            }
            Vertex::Surface(ref v) => {
                if let Some(sampled_bsdf) =
                    v.its
                        .mesh
                        .bsdf
                        .sample(&v.its.uv, &v.its.wi, sampler.next2d())
                {
                    // Update the throughput
                    *throughput *= &sampled_bsdf.weight;
                    if throughput.is_zero() {
                        return (None, None);
                    }

                    // Check RR
                    let rr_weight = throughput.channel_max().min(0.95);
                    if rr_weight < sampler.next() {
                        return (None, None);
                    }
                    let rr_weight = 1.0 / rr_weight;
                    throughput.scale(rr_weight);

                    // Generate the new ray and do the intersection
                    let d_out_global = v.its.frame.to_world(sampled_bsdf.d);
                    let ray = Ray::new(v.its.p, d_out_global);
                    let (edge, new_vertex) = Edge::from_ray(
                        path,
                        &ray,
                        vertex_id,
                        sampled_bsdf.pdf.clone(),
                        sampled_bsdf.weight,
                        rr_weight,
                        accel,
                        scene,
                        id_strategy,
                    );
                    return (Some(edge), new_vertex);
                }

                (None, None)
            }
            Vertex::Light(ref v) => {
                // For now, just computing the outgoing direction
                // Using cosine base weighting as we know that the light source
                // can only be cosine based isotropic lighting
                let d_out = cosine_sample_hemisphere(sampler.next2d());
                if d_out.z == 0.0 {
                    return (None, None); // Failed to sample the outgoing direction
                }

                let frame = Frame::new(v.n);
                let d_out_global = frame.to_world(d_out);
                let ray = Ray::new(v.pos, d_out_global);
                let weight = Color::one(); // Perfectly importance sampled

                let (edge, new_vertex) = Edge::from_ray(
                    path,
                    &ray,
                    vertex_id,
                    PDF::SolidAngle(d_out.z * std::f32::consts::FRAC_1_PI),
                    weight,
                    1.0,
                    accel,
                    scene,
                    id_strategy,
                );

                (Some(edge), new_vertex)
            }
        }
    }
}
impl SamplingStrategy for DirectionalSamplingStrategy {
    fn sample<'scene, 'emitter>(
        &self,
        path: &mut Path<'scene, 'emitter>,
        vertex_id: VertexID,
        accel: &'scene Acceleration,
        scene: &'scene Scene,
        _emitters: &'emitter EmitterSampler,
        mut throughput: Color,
        sampler: &mut Sampler,
        id_strategy: usize,
    ) -> Option<(VertexID, Color)> {
        // Generate the next edge and the next vertex
        let (edge, new_vertex) = DirectionalSamplingStrategy::bounce(
            path,
            vertex_id,
            accel,
            scene,
            &mut throughput,
            sampler,
            id_strategy,
        );

        // Update the edge if we sucesfull sample it
        if let Some(e) = edge {
            match path.vertex_mut(vertex_id) {
                Vertex::Sensor(ref mut v) => {
                    v.edge_out = Some(e);
                }
                Vertex::Surface(ref mut v) => {
                    v.edge_out.push(e);
                }
                Vertex::Light(ref mut v) => {
                    v.edge_out = Some(e);
                }
            }
        }

        if let Some(new_vertex) = new_vertex {
            Some((new_vertex, throughput))
        } else {
            None
        }
    }
    fn pdf<'scene, 'emitter>(
        &self,
        path: &Path<'scene, 'emitter>,
        _scene: &'scene Scene,
        _emitters: &'emitter EmitterSampler,
        vertex_id: VertexID,
        edge_id: EdgeID,
    ) -> Option<f32> {
        let edge = path.edge(edge_id);
        if !edge.next_on_light_source(path) {
            return None;
        }

        match path.vertex(vertex_id) {
            Vertex::Surface(ref v) => {
                if v.its.mesh.bsdf.is_smooth() {
                    return None;
                }
                if let PDF::SolidAngle(pdf) = v.its.mesh.bsdf.pdf(
                    &v.its.uv,
                    &v.its.wi,
                    &v.its.frame.to_local(edge.d),
                    Domain::SolidAngle,
                ) {
                    return Some(pdf);
                }
                unimplemented!();
            }
            Vertex::Sensor(ref _v) => Some(1.0),
            _ => None,
        }
    }
}

pub struct LightSamplingStrategy {}
impl SamplingStrategy for LightSamplingStrategy {
    fn sample<'scene, 'emitter>(
        &self,
        path: &mut Path<'scene, 'emitter>,
        vertex_id: VertexID,
        accel: &'scene Acceleration,
        _scene: &'scene Scene,
        emitters: &'emitter EmitterSampler,
        _throughput: Color,
        sampler: &mut Sampler,
        id_strategy: usize,
    ) -> Option<(VertexID, Color)> {
        let (edge, _next_vertex) = match path.vertex(vertex_id) {
            Vertex::Surface(ref v) => {
                if v.its.mesh.bsdf.is_smooth() {
                    return None;
                }

                // Generate the light sampling record based on the current vertex location
                // Note that during this procedure, we did not evaluate the product of the path throughput
                // and the incomming direct light. This evaluation will be done later when MIS
                // will be computed.
                let light_record = emitters.sample_light(
                    &v.its.p,
                    sampler.next(),
                    sampler.next(),
                    sampler.next2d(),
                );
                let d_out_local = v.its.frame.to_local(light_record.d);
                let visible = accel.visible(&v.its.p, &light_record.p);
                if light_record.is_valid() && d_out_local.z > 0.0 && visible {
                    let next_vertex = Vertex::Light(EmitterVertex {
                        pos: light_record.p,
                        n: light_record.n,
                        emitter: light_record.emitter,
                        edge_in: None,
                        edge_out: None,
                    });

                    // FIXME: Only work for diffuse light
                    let mut weight = light_record.weight;
                    let emission = light_record.emitter.emitted_luminance(-d_out_local);
                    weight.r /= emission.r;
                    weight.g /= emission.g;
                    weight.b /= emission.b;

                    // Need to evaluate the BSDF
                    weight *= &v.its.mesh.bsdf.eval(
                        &v.its.uv,
                        &v.its.wi,
                        &d_out_local,
                        Domain::SolidAngle,
                    );

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
            Vertex::Surface(ref mut v) => {
                v.edge_out.push(edge);
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
        let edge = path.edge(edge_id);
        if !edge.next_on_light_source(path) {
            return None;
        }

        let vertex = path.vertex(vertex_id);
        match vertex {
            Vertex::Surface(ref v) => {
                // Impossible to sample from a Dirac distribution
                if v.its.mesh.bsdf.is_smooth() {
                    return None;
                }
                // Know the the light is intersectable so have a solid angle PDF
                let ray = Ray::new(vertex.position(), edge.d);
                if let Some(next_vertex_id) = edge.vertices.1 {
                    match path.vertex(next_vertex_id) {
                        Vertex::Surface(ref v) => {
                            if let PDF::SolidAngle(light_pdf) = emitters
                                .direct_pdf(v.its.mesh, &LightSamplingPDF::new(&ray, &v.its))
                            {
                                return Some(light_pdf);
                            }
                        }
                        Vertex::Light(ref v) => {
                            if let PDF::SolidAngle(light_pdf) = emitters.direct_pdf(
                                v.emitter,
                                &LightSamplingPDF {
                                    o: ray.o,
                                    p: v.pos,
                                    n: v.n,
                                    dir: ray.d,
                                },
                            ) {
                                return Some(light_pdf);
                            }
                        }
                        _ => return None,
                    }
                }
                None
            }
            _ => None,
        }
    }
}

pub fn generate<'scene, 'emitter, T: Technique>(
    path: &mut Path<'scene, 'emitter>,
    accel: &'scene Acceleration,
    scene: &'scene Scene,
    emitters: &'emitter EmitterSampler,
    sampler: &mut Sampler,
    technique: &mut T,
) -> Vec<(VertexID, Color)> {
    let root = technique.init(path, accel, scene, sampler, emitters);
    let mut curr = root.clone();
    let mut next = vec![];
    let mut depth = 1;
    while !curr.is_empty() {
        // Do a wavefront processing of the different vertices
        next.clear();
        for (curr_vertex_id, throughput) in &curr {
            // For all the sampling techniques
            // This is the continue if we want to continue or not
            // For example, we might want to not push the vertex if we have reach the depth limit
            if technique.expand(path.vertex(*curr_vertex_id), depth) {
                for (id_sampling, sampling) in technique
                    .strategies(path.vertex(*curr_vertex_id))
                    .iter()
                    .enumerate()
                {
                    // If we want to continue the tracing toward this direction
                    if let Some((new_vertex, new_throughput)) = sampling.sample(
                        path,
                        *curr_vertex_id,
                        accel,
                        scene,
                        emitters,
                        *throughput,
                        sampler,
                        id_sampling,
                    ) {
                        next.push((new_vertex, new_throughput));
                    }
                }
            }
        }
        // Flip-flap buffer
        mem::swap(&mut curr, &mut next);
        depth += 1;
    }

    root
}

pub trait Technique {
    fn init<'scene, 'emitter>(
        &mut self,
        path: &mut Path<'scene, 'emitter>,
        accel: &'scene Acceleration,
        scene: &'scene Scene,
        sampler: &mut Sampler,
        emitters: &'emitter EmitterSampler,
    ) -> Vec<(VertexID, Color)>;
    fn strategies(&self, vertex: &Vertex) -> &Vec<Box<SamplingStrategy>>;
    fn expand(&self, vertex: &Vertex, depth: u32) -> bool;
}
