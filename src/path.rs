use cgmath::*;
use material::*;
use sampler::*;
use scene::*;
use structure::*;

#[derive(Clone)]
pub struct Edge {
    pub dist: Option<f32>,
    pub d: Vector3<f32>,
}

#[derive(Clone)]
pub struct SensorVertex {
    pub uv: Point2<f32>,
    pub pos: Point3<f32>,
    // FIXME: Add as Option
    pub pdf: f32, // FIXME: Add as Option
}

#[derive(Clone)]
pub struct SurfaceVertex<'a> {
    pub its: Intersection<'a>,
    pub throughput: Color,
    pub sampled_bsdf: Option<SampledDirection>,
    pub rr_weight: f32,
}

#[derive(Clone)]
pub struct SurfaceVertexShift {
    /// Containts the throughput times the jacobian
    pub throughput: Color,
    /// Contains the ratio of PDF (with the Jacobian embeeded)
    pub pdf_ratio: f32,
}

#[derive(Clone)]
pub enum Vertex<'a> {
    Sensor(SensorVertex),
    Surface(SurfaceVertex<'a>),
    SurfaceShift(SurfaceVertexShift),
}

impl<'a> Vertex<'a> {
    pub fn new_sensor_vertex(uv: Point2<f32>, pos: Point3<f32>) -> Vertex<'a> {
        Vertex::Sensor(SensorVertex {
            uv,
            pos,
            pdf: 1.0,
        })
    }

    pub fn generate_next(&mut self,
                         scene: &'a Scene,
                         sampler: Option<&mut Sampler>) -> (Option<Edge>, Option<Vertex<'a>>) {
        match *self {
            Vertex::Sensor(ref mut v) => {
                let ray = scene.camera.generate(v.uv);
                let its = match scene.trace(&ray) {
                    Some(its) => its,
                    None => return (Some(Edge { dist: None, d: ray.d }), None),
                };

                (
                    Some(Edge { dist: Some(its.dist), d: ray.d }),
                    Some(Vertex::Surface(SurfaceVertex {
                        its: its,
                        throughput: Color::one(),
                        sampled_bsdf: None,
                        rr_weight: 1.0,
                    }))
                )
            }
            Vertex::Surface(ref mut v) => {
                assert!(!sampler.is_none());
                let sampler = sampler.unwrap();

                v.sampled_bsdf = match v.its.mesh.bsdf.sample(&v.its.wi, sampler.next2d()) {
                    Some(x) => Some(x),
                    None => return (None, None)
                };
                let sampled_bsdf = v.sampled_bsdf.as_ref().unwrap();

                // Update the throughput
                let mut new_throughput = v.throughput * sampled_bsdf.weight;
                if new_throughput.is_zero() {
                    return (None, None);
                }


                // Generate the new ray and do the intersection
                let d_out_global = v.its.frame.to_world(sampled_bsdf.d);
                let ray = Ray::new(v.its.p, d_out_global);
                let its = match scene.trace(&ray) {
                    Some(its) => its,
                    None => {
                        return (Some(Edge { dist: None, d: d_out_global }),
                                None);
                    }
                };

                // Check RR
                let rr_weight = new_throughput.channel_max().min(0.95);
                if rr_weight < sampler.next() {
                    return (Some(Edge {
                        dist: Some(its.dist),
                        d: d_out_global,
                    }), None);
                }
                new_throughput /= rr_weight;

                (Some(Edge {
                    dist: Some(its.dist),
                    d: d_out_global,
                }),
                 Some(Vertex::Surface(
                     SurfaceVertex {
                         its,
                         throughput: new_throughput,
                         sampled_bsdf: None,
                         rr_weight,
                     }
                 ))
                )
            }
            _ => panic!("Wrong vertex type"),
        }
    }
}

pub struct Path<'a> {
    pub vertices: Vec<Vertex<'a>>,
    pub edges: Vec<Edge>,
}


enum ShiftGeometricState {
    NotConnected,
    RecentlyConnected,
    Connected,
}

impl<'a> Path<'a> {
    pub fn from_sensor((ix, iy): (u32, u32),
                       scene: &'a Scene,
                       sampler: &mut Sampler,
                       max_depth: Option<u32>) -> Option<Path<'a>> {
        let pix = Point2::new(
            ix as f32 + sampler.next(),
            iy as f32 + sampler.next(),
        );
        let mut vertices = vec![Vertex::new_sensor_vertex(pix,
                                                          scene.camera.param.pos)
        ];
        let mut edges: Vec<Edge> = vec![];

        let mut depth = 1;
        while max_depth.map_or(true, |max| depth < max) {
            match vertices.last_mut().unwrap().generate_next(scene, Some(sampler)) {
                (Some(edge), Some(vertex)) => {
                    edges.push(edge);
                    vertices.push(vertex);
                }
                (Some(edge), None) => {
                    // This case model a path where we was able to generate a direction
                    // But somehow, not able to generate a intersection point, because:
                    //  - no geometry have been intersected
                    //  - russian roulette kill the path
                    edges.push(edge);
                    return Some(Path {
                        vertices,
                        edges,
                    });
                }
                _ => { // Kill for a lot of reason ...
                    return Some(Path {
                        vertices,
                        edges,
                    });
                }
            }
            depth += 1;
        }

        Some(Path {
            vertices,
            edges,
        })
    }

    // IDEA:
    // Add the jacobian value inside the flux and pdf to more easy MIS
    pub fn shift_geometric(&self, shift_pix: Point2<f32>, scene: &'a Scene) -> Option<Path<'a>> {
        // FIXME: Need to implement G-PT shift mapping
        // FIXME: The idea of this code is to shift the path geometry
        // FIXME: without evaluating the direct lighting (compared to G-PT)
        let mut v0 = Vertex::new_sensor_vertex(shift_pix,
                                               scene.camera.param.pos);
        let (e0, v1) = match v0.generate_next(scene, None) {
            (Some(e), Some(v)) => (e, v),
            _ => return None, // FIXME: This is not correct for now
        };

        let mut vertices = vec![v0, v1];
        let mut edges = vec![e0];
        let mut state = ShiftGeometricState::NotConnected;
        let mut pdf = 1.0;

        for i in 1..self.vertices.len() {
            match state {
                ShiftGeometricState::NotConnected => {
                    let main_currrent = match &self.vertices[i] {
                        &Vertex::Surface(ref x) => x,
                        _ => panic!("Wrong main_current vertex type"),
                    };
                    let main_bsdf_pdf = match main_currrent.sampled_bsdf.as_ref().unwrap().pdf {
                        PDF::SolidAngle(x) => x,
                        _ => panic!("main_bsdf_pdf is not in solid angle"),
                    };

                    match self.vertices.get(i + 1) {
                        //FIXME: Are we sure about that? Because the path might be not
                        //FIXME: Because a edge of the path can be missing
                        None => return Some(Path {
                            vertices,
                            edges,
                        }),
                        Some(&Vertex::Surface(ref main_next)) => {
                            let new_vertex = {
                                let main_edge = &self.edges[i - 1];
                                let shift_current = match vertices.last().unwrap() {
                                    &Vertex::Surface(ref x) => x,
                                    _ => panic!("Un-expected path for the shift mapping"), // If we have something else, panic!
                                };
                                // Check the visibility
                                if !scene.visible(&shift_current.its.p, &main_next.its.p) {
                                    // Just return now the shift path is dead due to visibility
                                    return None;
                                }

                                // Compute the new direction for evaluating the BSDF
                                let mut shift_d_out_global = (main_next.its.p - shift_current.its.p);
                                let shift_distance = shift_d_out_global.magnitude();
                                shift_d_out_global /= shift_distance;
                                let shift_d_out_local = shift_current.its.frame.to_local(shift_d_out_global);
                                // BSDF shift path
                                let shift_bsdf_value = shift_current.its.mesh.bsdf.eval(
                                    &shift_current.its.wi, &shift_d_out_local);
                                let shift_bsdf_pdf = match shift_current.its.mesh.bsdf.pdf(
                                    &shift_current.its.wi, &shift_d_out_local) {
                                    PDF::SolidAngle(x) => x,
                                    _ => panic!("shift_bsdf_pdf is not in Solid angle"),
                                };

                                if shift_bsdf_pdf == 0.0 || shift_bsdf_value.is_zero() {
                                    // Just return now the shift path as the rest of the vertex will be 0
                                    return None;
                                }
                                // Compute the Jacobian value
                                let jacobian = (main_next.its.n_g.dot(-shift_d_out_global) * main_next.its.dist.powi(2)).abs()
                                    / (main_next.its.n_g.dot(-main_edge.d) * (shift_current.its.p - main_next.its.p).magnitude2()).abs();
                                assert!(jacobian.is_finite());
                                assert!(jacobian >= 0.0);

                                Some((Vertex::SurfaceShift(
                                    SurfaceVertexShift {
                                        throughput: shift_current.throughput * &(shift_bsdf_value * (jacobian / main_bsdf_pdf)),
                                        pdf_ratio: pdf * (shift_bsdf_pdf * jacobian) / main_bsdf_pdf,
                                    }
                                ), Edge {
                                    dist: Some(shift_distance),
                                    d: shift_d_out_global,
                                }))
                            };

                            // Update shift path
                            match new_vertex {
                                None => return Some(Path {
                                    vertices,
                                    edges,
                                }),
                                Some((v, edge)) => {
                                    vertices.push(v);
                                    edges.push(edge);
                                }
                            }

                            // Change the state of the shift
                            state = ShiftGeometricState::RecentlyConnected;
                        }
                        _ => panic!("Encounter wrong vertex type"),
                    }
                }
                ShiftGeometricState::RecentlyConnected => {

                }
                ShiftGeometricState::Connected => {
                    match &self.vertices.get(i) {
                        &None => return Some(Path {
                            vertices,
                            edges,
                        }),
                        &Some(&Vertex::Surface(ref main_next)) => {
                            match &main_next.sampled_bsdf {
                                &Some(ref x) => {
                                    let new_vertex = {
                                        let shift_current = match vertices.last().unwrap() {
                                            &Vertex::SurfaceShift(ref x) => x,
                                            _ => panic!("Un-expected path for the shift mapping"), // If we have something else, panic!
                                        };
                                        Vertex::SurfaceShift(
                                            SurfaceVertexShift {
                                                throughput: x.weight * shift_current.throughput,
                                                pdf_ratio: shift_current.pdf_ratio, // No change here
                                            }
                                        )
                                    };
                                    // Just recopy the path
                                    vertices.push(new_vertex);
                                    edges.push(self.edges[i-1].clone());
                                }
                                _ => {
                                    // The main path is dead, stop doing the shift
                                    // FIXME: Maybe one vertex will miss in this case
                                    return Some(Path {
                                        vertices,
                                        edges,
                                    });
                                }
                            }
                        }
                        _ => panic!("Encounter wrong vertex type"),
                    }
                }
            }
        }

        Some(Path {
            vertices,
            edges,
        })
    }
}
