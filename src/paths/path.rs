use cgmath::*;
use paths::vertex::*;
use samplers::*;
use scene::*;
use std::cell::RefCell;
use std::iter::Sum;
use std::mem;
use std::rc::Rc;
use structure::*;
use Scale;

pub struct Path<'a> {
    pub root: Rc<VertexPtr<'a>>,
}

pub trait SamplingStrategy<S: Sampler> {
    fn sample<'a>(
        &self,
        vertex: Rc<VertexPtr<'a>>,
        scene: &'a Scene,
        throughput: Color,
        sampler: &mut S,
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
    pub fn bounce<'a, S: Sampler>(
        vertex: &Rc<VertexPtr<'a>>,
        scene: &'a Scene,
        throughput: &mut Color,
        sampler: &mut S,
    ) -> (Option<Rc<EdgePtr<'a>>>, Option<Rc<VertexPtr<'a>>>) {
        match *vertex.borrow() {
            Vertex::Sensor(ref v) => {
                let ray = scene.camera.generate(v.uv);
                let (edge, new_vertex) =
                    Edge::from_ray(ray, &vertex, PDF::SolidAngle(1.0), Color::one(), 1.0, scene);
                return (Some(edge), new_vertex);
            }
            Vertex::Surface(ref v) => {
                if let Some(sampled_bsdf) = v.its.mesh.bsdf.sample(
                    &v.its.uv,
                    &v.its.wi,
                    sampler.next2d(),
                ) {
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
                        ray,
                        &vertex,
                        sampled_bsdf.pdf.clone(),
                        sampled_bsdf.weight,
                        rr_weight,
                        scene,
                    );
                    return (Some(edge), new_vertex);
                }

                return (None, None);
            }
            _ => unimplemented!(),
        }
    }
}
impl<S: Sampler> SamplingStrategy<S> for DirectionalSamplingStrategy {
    fn sample<'a>(
        &self,
        vertex: Rc<VertexPtr<'a>>,
        scene: &'a Scene,
        mut throughput: Color,
        sampler: &mut S,
    ) -> Option<(Rc<VertexPtr<'a>>, Color)> {
        // Generate the next edge and the next vertex
        let (edge, new_vertex) =
            DirectionalSamplingStrategy::bounce(&vertex, scene, &mut throughput, sampler);

        // Update the edge if we sucesfull sample it
        if let Some(e) = edge {
            match *vertex.borrow_mut() {
                Vertex::Sensor(ref mut v) => {
                    v.edge = Some(e);
                }
                Vertex::Surface(ref mut v) => {
                    v.edge_out.push(e);
                }
                _ => unimplemented!(),
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
            Vertex::Sensor(ref v) => return Some(1.0),
            _ => return None,
        }
    }
}

pub struct LightSamplingStrategy {}
impl<S: Sampler> SamplingStrategy<S> for LightSamplingStrategy {
    fn sample<'a>(
        &self,
        vertex: Rc<VertexPtr<'a>>,
        scene: &'a Scene,
        mut throughput: Color,
        sampler: &mut S,
    ) -> Option<(Rc<VertexPtr<'a>>, Color)> {
        let (edge, next_vertex) = match *vertex.borrow() {
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
                        edge: None,
                    })));

                    // FIXME: Only work for diffuse light
                    let mut weight = light_record.weight;
                    weight.r /= light_record.emitter.emission.r;
                    weight.g /= light_record.emitter.emission.g;
                    weight.b /= light_record.emitter.emission.b;
                    (
                        Edge::from_vertex(&vertex, light_record.pdf, weight, 1.0, &next_vertex),
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
                                scene.direct_pdf(LightSamplingPDF::new(&ray, &v.its))
                            {
                                return Some(light_pdf);
                            }
                        }
                        Vertex::Emitter(ref v) => {
                            if let PDF::SolidAngle(light_pdf) = scene.direct_pdf(LightSamplingPDF {
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
                return None;
            }
            _ => return None,
        }
    }
}

impl<'a> Path<'a> {
    pub fn from_sensor<S: Sampler>(
        (ix, iy): (u32, u32),
        scene: &'a Scene,
        sampler: &mut S,
        max_depth: Option<u32>,
        samplings: &Vec<Box<SamplingStrategy<S>>>,
    ) -> Option<Path<'a>> {
        // Initialize the root not
        let root = Rc::new(RefCell::new(Vertex::Sensor(SensorVertex {
            uv: Point2::new(ix as f32 + sampler.next(), iy as f32 + sampler.next()),
            pos: scene.camera.param.pos.clone(),
            edge: None,
        })));

        let mut curr = vec![(root.clone(), Color::one())];
        let mut next = vec![];
        let mut depth = 1;

        while max_depth.map_or(true, |max| depth < max) {
            if curr.is_empty() {
                break;
            }

            next.clear();
            for (c, t) in &curr {
                for sampling in samplings {
                    if let Some((v, c)) = sampling.sample(c.clone(), scene, *t, sampler) {
                        next.push((v, c));
                    }
                }
            }
            mem::swap(&mut curr, &mut next);
            depth += 1;
        }

        Some(Path { root })
    }

    fn evaluate_vertex<S: Sampler>(
        scene: &'a Scene,
        vertex: &Rc<VertexPtr<'a>>,
        samplings: &Vec<Box<SamplingStrategy<S>>>,
    ) -> Color {
        let mut l_i = Color::zero();
        match *vertex.borrow() {
            Vertex::Surface(ref v) => {
                for (i, edge) in v.edge_out.iter().enumerate() {
                    let contrib = edge.borrow().contribution();
                    if !contrib.is_zero() && i == 1 {
                        // let weight = if let PDF::SolidAngle(v) = edge.borrow().pdf_direction {
                        //     let total: f32 = samplings
                        //         .iter()
                        //         .map(|s| {
                        //             if let Some(v) = s.pdf(scene, &vertex, edge) {
                        //                 v
                        //             } else {
                        //                 0.0
                        //             }
                        //         })
                        //         .sum();
                        //     v / total
                        // } else {
                        //     1.0
                        // };
                        // let weight = 1.0 / v.edge_out.len() as f32;
                        let weight = 1.0;
                        l_i += contrib * weight;
                    }

                    let edge = edge.borrow();
                    if let Some(ref vertex_next) = edge.vertices.1 {
                        l_i += edge.weight
                            * edge.rr_weight
                            * Path::evaluate_vertex(scene, &vertex_next, samplings);
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
                    l_i += edge.borrow().weight
                        * Path::evaluate_vertex(scene, &vertex_next, samplings);
                }
            }
            _ => {}
        };

        return l_i;
    }
    pub fn evaluate<S: Sampler>(
        &self,
        scene: &'a Scene,
        samplings: &Vec<Box<SamplingStrategy<S>>>,
    ) -> Color {
        return Path::evaluate_vertex(scene, &self.root, samplings);
    }
}
