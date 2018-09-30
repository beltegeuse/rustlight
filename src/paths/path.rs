use math::{cosine_sample_hemisphere, Frame};
use paths::vertex::*;
use samplers::*;
use scene::*;
use std;
use std::cell::RefCell;
use std::mem;
use std::rc::Rc;
use structure::*;
use Scale;

pub trait SamplingStrategy {
    fn sample<'a>(
        &self,
        vertex: Rc<VertexPtr<'a>>,
        scene: &'a Scene,
        throughput: Color,
        sampler: &mut Sampler,
        id_strategy: usize,
    ) -> Option<(Rc<VertexPtr<'a>>, Color)>;

    // All PDF have to be inside the same domain
    fn pdf<'a>(
        &self,
        scene: &'a Scene,
        vertex: &Rc<VertexPtr<'a>>,
        edge: &Rc<EdgePtr<'a>>,
    ) -> Option<f32>;
}

pub struct DirectionalSamplingStrategy {}
impl DirectionalSamplingStrategy {
    pub fn bounce<'a>(
        vertex: &Rc<VertexPtr<'a>>,
        scene: &'a Scene,
        throughput: &mut Color,
        sampler: &mut Sampler,
        id_strategy: usize,
    ) -> (Option<Rc<EdgePtr<'a>>>, Option<Rc<VertexPtr<'a>>>) {
        match *vertex.borrow() {
            Vertex::Sensor(ref v) => {
                let ray = scene.camera.generate(v.uv);
                let (edge, new_vertex) = Edge::from_ray(
                    &ray,
                    &vertex,
                    PDF::SolidAngle(1.0),
                    Color::one(),
                    1.0,
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
                        &ray,
                        &vertex,
                        sampled_bsdf.pdf.clone(),
                        sampled_bsdf.weight,
                        rr_weight,
                        scene,
                        id_strategy,
                    );
                    return (Some(edge), new_vertex);
                }

                (None, None)
            }
            Vertex::Emitter(ref v) => {
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
                    &ray,
                    &vertex,
                    PDF::SolidAngle(d_out.z * std::f32::consts::FRAC_1_PI),
                    weight,
                    1.0,
                    scene,
                    id_strategy,
                );

                (Some(edge), new_vertex)
            }
        }
    }
}
impl SamplingStrategy for DirectionalSamplingStrategy {
    fn sample<'a>(
        &self,
        vertex: Rc<VertexPtr<'a>>,
        scene: &'a Scene,
        mut throughput: Color,
        sampler: &mut Sampler,
        id_strategy: usize,
    ) -> Option<(Rc<VertexPtr<'a>>, Color)> {
        // Generate the next edge and the next vertex
        let (edge, new_vertex) = DirectionalSamplingStrategy::bounce(
            &vertex,
            scene,
            &mut throughput,
            sampler,
            id_strategy,
        );

        // Update the edge if we sucesfull sample it
        if let Some(e) = edge {
            match *vertex.borrow_mut() {
                Vertex::Sensor(ref mut v) => {
                    v.edge_out = Some(e);
                }
                Vertex::Surface(ref mut v) => {
                    v.edge_out.push(e);
                }
                Vertex::Emitter(ref mut v) => {
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
    fn pdf<'a>(
        &self,
        _scene: &'a Scene,
        vertex: &Rc<VertexPtr<'a>>,
        edge: &Rc<EdgePtr<'a>>,
    ) -> Option<f32> {
        if !edge.borrow().next_on_light_source() {
            return None;
        }

        match *vertex.borrow() {
            Vertex::Surface(ref v) => {
                if let PDF::SolidAngle(pdf) = v.its.mesh.bsdf.pdf(
                    &v.its.uv,
                    &v.its.wi,
                    &v.its.frame.to_local(edge.borrow().d),
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
    fn sample<'a>(
        &self,
        vertex: Rc<VertexPtr<'a>>,
        scene: &'a Scene,
        mut _throughput: Color,
        sampler: &mut Sampler,
        id_strategy: usize,
    ) -> Option<(Rc<VertexPtr<'a>>, Color)> {
        let (edge, _next_vertex) = match *vertex.borrow() {
            Vertex::Surface(ref v) => {
                if v.its.mesh.bsdf.is_smooth() {
                    return None;
                }

                // Generate the light sampling record based on the current vertex location
                // Note that during this procedure, we did not evaluate the product of the path throughput
                // and the incomming direct light. This evaluation will be done later when MIS
                // will be computed.
                let light_record =
                    scene.sample_light(&v.its.p, sampler.next(), sampler.next(), sampler.next2d());
                let d_out_local = v.its.frame.to_local(light_record.d);
                let visible = scene.visible(&v.its.p, &light_record.p);
                if light_record.is_valid() && d_out_local.z > 0.0 && visible {
                    let next_vertex = Rc::new(RefCell::new(Vertex::Emitter(EmitterVertex {
                        pos: light_record.p,
                        n: light_record.n,
                        mesh: light_record.emitter,
                        edge_in: None,
                        edge_out: None,
                    })));

                    // FIXME: Only work for diffuse light
                    let mut weight = light_record.weight;
                    weight.r /= light_record.emitter.emission.r;
                    weight.g /= light_record.emitter.emission.g;
                    weight.b /= light_record.emitter.emission.b;

                    // Need to evaluate the BSDF
                    weight *= &v.its.mesh.bsdf.eval(&v.its.uv, &v.its.wi, &d_out_local);

                    (
                        Edge::from_vertex(
                            &vertex,
                            light_record.pdf,
                            weight,
                            1.0,
                            &next_vertex,
                            id_strategy,
                        ),
                        next_vertex,
                    )
                } else {
                    return None;
                }
            }
            _ => return None,
        };

        // Update the out edge
        match *vertex.borrow_mut() {
            Vertex::Surface(ref mut v) => {
                v.edge_out.push(edge.clone());
            }
            _ => unimplemented!(),
        }

        None // Finish the sampling here
    }

    fn pdf<'a>(
        &self,
        scene: &'a Scene,
        vertex: &Rc<VertexPtr<'a>>,
        edge: &Rc<EdgePtr<'a>>,
    ) -> Option<f32> {
        if !edge.borrow().next_on_light_source() {
            return None;
        }

        match *vertex.borrow() {
            Vertex::Surface(ref v) => {
                // Impossible to sample from a Dirac distribution
                if v.its.mesh.bsdf.is_smooth() {
                    return None;
                }
                // Know the the light is intersectable so have a solid angle PDF
                let ray = Ray::new(vertex.borrow().position(), edge.borrow().d);
                if let Some(ref next_vertex) = edge.borrow().vertices.1 {
                    match *next_vertex.borrow() {
                        Vertex::Surface(ref v) => {
                            if let PDF::SolidAngle(light_pdf) =
                                scene.direct_pdf(&LightSamplingPDF::new(&ray, &v.its))
                            {
                                return Some(light_pdf);
                            }
                        }
                        Vertex::Emitter(ref v) => {
                            if let PDF::SolidAngle(light_pdf) =
                                scene.direct_pdf(&LightSamplingPDF {
                                    mesh: v.mesh,
                                    o: ray.o,
                                    p: v.pos,
                                    n: v.n,
                                    dir: ray.d,
                                }) {
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

pub fn generate<'a, T: Technique<'a>>(
    scene: &'a Scene,
    sampler: &mut Sampler,
    technique: &mut T,
) -> Vec<(Rc<RefCell<Vertex<'a>>>, Color)> {
    let root = technique.init(scene, sampler);
    let mut curr = root.clone();
    let mut next = vec![];
    let mut depth = 1;
    while !curr.is_empty() {
        // Do a wavefront processing of the different vertices
        next.clear();
        for (curr_vertex, throughput) in &curr {
            // For all the sampling techniques
            // This is the continue if we want to continue or not
            // For example, we might want to not push the vertex if we have reach the depth limit
            if technique.expand(&curr_vertex, depth) {
                for (id_sampling, sampling) in technique.strategies(&curr_vertex).iter().enumerate()
                {
                    // If we want to continue the tracing toward this direction
                    if let Some((new_vertex, new_throughput)) = sampling.sample(
                        curr_vertex.clone(),
                        scene,
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

pub trait Technique<'a> {
    fn init(
        &mut self,
        scene: &'a Scene,
        sampler: &mut Sampler,
    ) -> Vec<(Rc<RefCell<Vertex<'a>>>, Color)>;
    fn strategies(&self, vertex: &Rc<RefCell<Vertex<'a>>>) -> &Vec<Box<SamplingStrategy>>;
    fn expand(&self, vertex: &Rc<RefCell<Vertex<'a>>>, depth: u32) -> bool;
}
