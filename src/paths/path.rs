use cgmath::*;
use paths::vertex::*;
use samplers::*;
use scene::*;
use structure::*;
use std::rc::Rc;
use std::cell::RefCell;
use Scale;
use std::mem;

/// Power heuristic for path tracing or direct lighting
pub fn mis_weight(pdf_a: f32, pdf_b: f32) -> f32 {
    if pdf_a == 0.0 {
        warn!("MIS weight requested for 0.0 pdf");
        return 0.0;
    }
    assert!(pdf_a.is_finite());
    assert!(pdf_b.is_finite());
    let w = pdf_a.powi(2) / (pdf_a.powi(2) + pdf_b.powi(2));
    if w.is_finite() {
        w
    } else {
        warn!("Not finite MIS weight for: {} and {}", pdf_a, pdf_b);
        0.0
    }
}

pub struct Path<'a> {
    pub root: Rc<VertexPtr<'a>>,
}

pub trait SamplingStrategy<S: Sampler> {
    fn sample<'a>(
        &self, 
        vertex: Rc<VertexPtr<'a>>, 
        scene: &'a Scene, 
        throughput: Color,
        sampler: &mut S) -> Option<(Rc<VertexPtr<'a>>, Color)>;

    // All PDF have to be inside the same domain
    fn pdf<'a>(&self, scene: &'a Scene, vertex: Rc<VertexPtr<'a>>, edge: Rc<EdgePtr<'a>>) -> Option<f32>;
}

pub struct DirectionalSamplingStrategy {
}
impl DirectionalSamplingStrategy {
    pub fn bounce<'a, S: Sampler>(vertex: &Rc<VertexPtr<'a>>, scene: &'a Scene,
        throughput: &mut Color,
        sampler: &mut S) -> (Option<Rc<EdgePtr<'a>>>, Option<Rc<VertexPtr<'a>>>) {
        match *vertex.borrow() {
            Vertex::Sensor(ref v) => {
                let ray = scene.camera.generate(v.uv);
                let (edge, new_vertex) = Edge::from_ray(ray, &vertex, PDF::SolidAngle(1.0), Color::one(), 1.0, scene);
                return (Some(edge), new_vertex);
            }
            Vertex::Surface(ref v) => {
                if let Some(sampled_bsdf) = v.its.mesh.bsdf.sample(
                    &v.its.uv,
                    &v.its.wi,
                    sampler.next2d()) {

                    // Update the throughput
                    *throughput *= &sampled_bsdf.weight;
                    if throughput.is_zero() {
                        return (None, None);
                    }

                    // Check RR
                    let rr_weight = throughput.channel_max().min(0.95);
                    if rr_weight < sampler.next() {
                        return (
                            None,
                            None,
                        );
                    }
                    let rr_weight = 1.0 / rr_weight;
                    throughput.scale(rr_weight);

                    // Generate the new ray and do the intersection
                    let d_out_global = v.its.frame.to_world(sampled_bsdf.d);
                    let ray = Ray::new(v.its.p, d_out_global);
                    let (edge, new_vertex) = Edge::from_ray(ray, &vertex, sampled_bsdf.pdf.clone(), sampled_bsdf.weight, rr_weight, scene);
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
        vertex: Rc<VertexPtr<'a>>, scene: &'a Scene,
        mut throughput: Color,
        sampler: &mut S) -> Option<(Rc<VertexPtr<'a>>, Color)> {
        // Generate the next edge and the next vertex
        let (edge, new_vertex) = DirectionalSamplingStrategy::bounce(&vertex, scene, &mut throughput, sampler);
 
        // Update the edge if we sucesfull sample it
        if let Some(e) = edge {
            match *vertex.borrow_mut() {
                 Vertex::Sensor(ref mut v) => {
                     v.edge = Some(e);
                 },
                 Vertex::Surface(ref mut v) => {
                     v.edge_out.push(e);
                 },
                 _ => unimplemented!(),
            }
        }

        if let Some(new_vertex) = new_vertex {
            Some((new_vertex, throughput))
        } else {
            None
        }
    } 
    fn pdf<'a>(&self, _scene: &'a Scene, vertex: Rc<VertexPtr<'a>>, edge: Rc<EdgePtr<'a>>) -> Option<f32> {
        match *vertex.borrow() {
         Vertex::Surface(ref v) => { 
             if let PDF::SolidAngle(pdf) = v.its.mesh.bsdf.pdf(
                    &v.its.uv,
                    &v.its.wi,
                    &v.its.frame.to_local(edge.borrow().d)) {
                        return Some(pdf);
                    }
                    unimplemented!();
                     },
                    _ => return None,
        }
    } 
}

pub struct LightSamplingStrategy {
}
impl<S: Sampler> SamplingStrategy<S> for LightSamplingStrategy {
    fn sample<'a>(
        &self,
        vertex: Rc<VertexPtr<'a>>, scene: &'a Scene,
        mut throughput: Color,
        sampler: &mut S) -> Option<(Rc<VertexPtr<'a>>, Color)> {

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
                    (Edge::from_vertex(&vertex, light_record.pdf, light_record.weight, 1.0, &next_vertex), next_vertex)
                } else {
                    return None;
                }
            },
            _ => return None,
        };

        // Update the out edge
        match *vertex.borrow_mut() {
            Vertex::Surface(ref mut v) => {
                v.edge_out.push(edge.clone());
            },
            _ => unimplemented!(),
        }
        
        None // Finish the sampling here
    }
    fn pdf<'a>(&self, scene: &'a Scene, vertex: Rc<VertexPtr<'a>>, edge: Rc<EdgePtr<'a>>) -> Option<f32> {
        match *vertex.borrow() {
            Vertex::Surface(ref v) => { 
                // Know the the light is intersectable so have a solid angle PDF
                let ray = Ray::new(vertex.borrow().position(), edge.borrow().d);
                if let PDF::SolidAngle(light_pdf) = scene.direct_pdf(LightSamplingPDF::new(&ray, &v.its)) {
                    return Some(light_pdf);
                }
                unimplemented!();
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
            for (c,t) in &curr {
                for sampling in samplings {
                    if let Some((v,c)) = sampling.sample(c.clone(), scene, *t, sampler) {
                        next.push((v,c));
                    }
                }
            }
            mem::swap(&mut curr, &mut next);
            depth += 1;
        }

        Some(Path { root })
    }

    fn evaluate_vertex<S: Sampler>(vertex: &VertexPtr<'a>, samplings: &Vec<Box<SamplingStrategy<S>>>) -> Color {
        let mut l_i = Color::zero();
        match *vertex.borrow() {
            Vertex::Surface(ref v) => {
                let edge = v.edge_in.upgrade().unwrap();
                if v.its.n_s.dot(-edge.borrow().d) >= 0.0 {
                    l_i += &v.its.mesh.emission;
                }
                for edge in &v.edge_out {
                    let edge = edge.borrow();
                    if let Some(ref vertex_next) = edge.vertices.1 {
                        l_i += edge.weight * edge.rr_weight * Path::evaluate_vertex(&vertex_next, samplings); 
                    }
                }
            }
            Vertex::Sensor(ref v) => {
                // Only one strategy where...
                let edge = v.edge.as_ref().unwrap();
                if let Some(ref vertex_next) = edge.borrow().vertices.1 {
                    l_i += edge.borrow().weight * Path::evaluate_vertex(&vertex_next, samplings); 
                } 
            }
            _ => {}
        };

        return l_i;
    } 
    pub fn evaluate<S: Sampler>(&self, samplings: &Vec<Box<SamplingStrategy<S>>>) -> Color {
        return Path::evaluate_vertex(&self.root, samplings);
    }
}


// impl<'a> LightSamplingVertex<'a> {
//     pub fn generate<S: Sampler>(
//         scene: &'a Scene,
//         sampler: &mut S,
//         vertex: &Vertex<'a>,
//     ) -> Option<LightSamplingVertex<'a>> {
//         match vertex {
//             &Vertex::Surface(ref v) => {
//                 // Check if the BSDF on the surface is smooth.
//                 // If it is the case, it is not useful to sample the direct lighting
//                 if v.its.mesh.bsdf.is_smooth() {
//                     return None;
//                 }

//                 // Generate the light sampling record based on the current vertex location
//                 // Note that during this procedure, we did not evaluate the product of the path throughput
//                 // and the incomming direct light. This evaluation will be done later when MIS
//                 // will be computed.
//                 let light_record =
//                     scene.sample_light(&v.its.p, sampler.next(), sampler.next(), sampler.next2d());
//                 let d_out_local = v.its.frame.to_local(light_record.d);
//                 let visible = scene.visible(&v.its.p, &light_record.p);
//                 if light_record.is_valid() && d_out_local.z > 0.0 {
//                     return Some(LightSamplingVertex {
//                         visible,
//                         sample: light_record,
//                     });
//                 } else {
//                     return None;
//                 }
//             }
//             _ => None,
//         }
//     }
// }

// pub struct PathWithDirect<'a> {
//     pub vertices: Vec<Vertex<'a>>,
//     pub edges: Vec<Edge>,
//     pub direct: Vec<Option<LightSamplingVertex<'a>>>,
// }

// impl<'a> PathWithDirect<'a> {
//     /// Generates the direct sampling record and evalutate the visibility
//     /// After PathWithDirect generated, it is possible to evaluate the total
//     /// contribution
//     pub fn generate<S: Sampler>(
//         scene: &'a Scene,
//         sampler: &mut S,
//         path: Path<'a>,
//         max_depth: Option<u32>,
//     ) -> PathWithDirect<'a> {
//         let mut direct = vec![];
//         for (i, vertex) in path.vertices.iter().enumerate() {
//             let light_sampling_vertex = match max_depth {
//                 Some(m) => {
//                     if i >= m as usize {
//                         None
//                     } else {
//                         LightSamplingVertex::generate(scene, sampler, &vertex)
//                     }
//                 }
//                 None => LightSamplingVertex::generate(scene, sampler, &vertex),
//             };
//             direct.push(light_sampling_vertex);
//         }
//         PathWithDirect {
//             vertices: path.vertices,
//             edges: path.edges,
//             direct,
//         }
//     }

//     pub fn evaluate(&self, scene: &'a Scene) -> Color {
//         let mut l_i = Color::zero();
//         let mut throughput = Color::one();
//         for (i, vertex) in self.vertices.iter().enumerate() {
//             match vertex {
//                 &Vertex::Surface(ref v) => {
//                     if i == 1 {
//                         l_i += throughput * (&v.its.mesh.emission);
//                     } else {
//                         // Direct lighting
//                         match &self.direct[i] {
//                             &Some(ref d) => {
//                                 if (d.visible) {
//                                     let light_pdf = match d.sample.pdf {
//                                         PDF::SolidAngle(v) => v,
//                                         _ => {
//                                             panic!("Unsupported light pdf type for pdf connection.")
//                                         }
//                                     };
//                                     let d_out_local = v.its.frame.to_local(d.sample.d);

//                                     if let PDF::SolidAngle(pdf_bsdf) =
//                                         v.its.mesh.bsdf.pdf(&v.its.uv, &v.its.wi, &d_out_local)
//                                     {
//                                         // Compute MIS weights
//                                         let weight_light = mis_weight(light_pdf, pdf_bsdf);
//                                         l_i += weight_light
//                                             * throughput
//                                             * v.its.mesh.bsdf.eval(
//                                                 &v.its.uv,
//                                                 &v.its.wi,
//                                                 &d_out_local,
//                                             )
//                                             * d.sample.weight;
//                                     }
//                                 }
//                             }
//                             _ => {}
//                         }

//                         // The BSDF sampling
//                         if v.its.mesh.is_light() && v.its.cos_theta() > 0.0 {
//                             let (pred_vertex_pos, pred_vertex_pdf) = match &self.vertices[i - 1] {
//                                 &Vertex::Surface(ref v) => {
//                                     (v.its.p, &v.sampled_bsdf.as_ref().unwrap().pdf)
//                                 }
//                                 _ => panic!("Wrong vertex type"),
//                             };

//                             let weight_bsdf = match pred_vertex_pdf {
//                                 &PDF::SolidAngle(pdf) => {
//                                     // As we have intersected the light, the PDF need to be in SA
//                                     let light_pdf = scene.direct_pdf(LightSamplingPDF {
//                                         mesh: v.its.mesh,
//                                         o: pred_vertex_pos,
//                                         p: v.its.p,
//                                         n: v.its.n_g, // FIXME: Geometrical normal?
//                                         dir: self.edges[i - 1].d,
//                                     });

//                                     mis_weight(pdf, light_pdf.value())
//                                 }
//                                 &PDF::Discrete(_v) => 1.0,
//                                 _ => panic!("Uncovered case"),
//                             };

//                             l_i += throughput * (&v.its.mesh.emission) * weight_bsdf;
//                         }

//                         throughput *= &v.sampled_bsdf.as_ref().unwrap().weight;
//                         throughput.scale(v.rr_weight);
//                     }
//                 }
//                 _ => {}
//             }
//         }
//         l_i
//     }
// }
