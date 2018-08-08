use cgmath::*;
use paths::vertex::*;
use samplers::*;
use scene::*;
use structure::*;

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
    pub vertices: Vec<Vertex<'a>>,
    pub edges: Vec<Edge>,
}

impl<'a> Path<'a> {
    pub fn from_sensor<S: Sampler>(
        (ix, iy): (u32, u32),
        scene: &'a Scene,
        sampler: &mut S,
        max_depth: Option<u32>,
    ) -> Option<Path<'a>> {
        let pix = Point2::new(ix as f32 + sampler.next(), iy as f32 + sampler.next());
        let mut vertices = vec![Vertex::new_sensor_vertex(pix, scene.camera.param.pos)];
        let mut edges: Vec<Edge> = vec![];

        let mut depth = 1;
        while max_depth.map_or(true, |max| depth < max) {
            match vertices
                .last_mut()
                .unwrap()
                .generate_next(scene, Some(sampler))
            {
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
                    return Some(Path { vertices, edges });
                }
                _ => {
                    // Kill for a lot of reason ...
                    return Some(Path { vertices, edges });
                }
            }
            depth += 1;
        }

        Some(Path { vertices, edges })
    }

    pub fn get_img_position(&self) -> Point2<f32> {
        match &self.vertices[0] {
            &Vertex::Sensor(ref v) => v.uv,
            _ => panic!("Impossible to gather the base path image position"),
        }
    }

    pub fn evaluate(&self) -> Color {
        let mut l_i = Color::zero();
        for vertex in &self.vertices {
            match vertex {
                Vertex::Surface(v) => {
                    l_i += v.throughput * (&v.its.mesh.emission);
                }
                _ => {}
            };
        }
        l_i
    }
}

pub struct LightSamplingVertex<'a> {
    pub visible: bool,
    pub sample: LightSampling<'a>,
}

impl<'a> LightSamplingVertex<'a> {
    pub fn generate<S: Sampler>(
        scene: &'a Scene,
        sampler: &mut S,
        vertex: &Vertex<'a>,
    ) -> Option<LightSamplingVertex<'a>> {
        match vertex {
            &Vertex::Surface(ref v) => {
                // Check if the BSDF on the surface is smooth.
                // If it is the case, it is not useful to sample the direct lighting
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
                if light_record.is_valid() && d_out_local.z > 0.0 {
                    return Some(LightSamplingVertex {
                        visible,
                        sample: light_record,
                    });
                } else {
                    return None;
                }
            }
            _ => None,
        }
    }
}

pub struct PathWithDirect<'a> {
    pub vertices: Vec<Vertex<'a>>,
    pub edges: Vec<Edge>,
    pub direct: Vec<Option<LightSamplingVertex<'a>>>,
}

impl<'a> PathWithDirect<'a> {
    /// Generates the direct sampling record and evalutate the visibility
    /// After PathWithDirect generated, it is possible to evaluate the total
    /// contribution
    pub fn generate<S: Sampler>(
        scene: &'a Scene,
        sampler: &mut S,
        path: Path<'a>,
        max_depth: Option<u32>,
    ) -> PathWithDirect<'a> {
        let mut direct = vec![];
        for (i, vertex) in path.vertices.iter().enumerate() {
            let light_sampling_vertex = match max_depth {
                Some(m) => {
                    if i >= m as usize {
                        None
                    } else {
                        LightSamplingVertex::generate(scene, sampler, &vertex)
                    }
                }
                None => LightSamplingVertex::generate(scene, sampler, &vertex),
            };
            direct.push(light_sampling_vertex);
        }
        PathWithDirect {
            vertices: path.vertices,
            edges: path.edges,
            direct,
        }
    }

    pub fn evaluate(&self, scene: &'a Scene) -> Color {
        let mut l_i = Color::zero();
        for (i, vertex) in self.vertices.iter().enumerate() {
            match vertex {
                &Vertex::Surface(ref v) => {
                    if i == 1 {
                        l_i += v.throughput * (&v.its.mesh.emission);
                    } else {
                        // Direct lighting
                        match &self.direct[i] {
                            &Some(ref d) => {
                                if (d.visible) {
                                    let light_pdf = match d.sample.pdf {
                                        PDF::SolidAngle(v) => v,
                                        _ => {
                                            panic!("Unsupported light pdf type for pdf connection.")
                                        }
                                    };
                                    let d_out_local = v.its.frame.to_local(d.sample.d);

                                    if let PDF::SolidAngle(pdf_bsdf) =
                                        v.its.mesh.bsdf.pdf(&v.its.uv, &v.its.wi, &d_out_local)
                                    {
                                        // Compute MIS weights
                                        let weight_light = mis_weight(light_pdf, pdf_bsdf);
                                        l_i += weight_light
                                            * v.throughput
                                            * v.its.mesh.bsdf.eval(
                                                &v.its.uv,
                                                &v.its.wi,
                                                &d_out_local,
                                            )
                                            * d.sample.weight;
                                    }
                                }
                            }
                            _ => {}
                        }

                        // The BSDF sampling
                        if v.its.mesh.is_light() && v.its.cos_theta() > 0.0 {
                            let (pred_vertex_pos, pred_vertex_pdf) = match &self.vertices[i - 1] {
                                &Vertex::Surface(ref v) => {
                                    (v.its.p, &v.sampled_bsdf.as_ref().unwrap().pdf)
                                }
                                _ => panic!("Wrong vertex type"),
                            };

                            let weight_bsdf = match pred_vertex_pdf {
                                &PDF::SolidAngle(pdf) => {
                                    // As we have intersected the light, the PDF need to be in SA
                                    let light_pdf = scene.direct_pdf(LightSamplingPDF {
                                        mesh: v.its.mesh,
                                        o: pred_vertex_pos,
                                        p: v.its.p,
                                        n: v.its.n_g, // FIXME: Geometrical normal?
                                        dir: self.edges[i - 1].d,
                                    });

                                    mis_weight(pdf, light_pdf.value())
                                }
                                &PDF::Discrete(_v) => 1.0,
                                _ => panic!("Uncovered case"),
                            };

                            l_i += v.throughput * (&v.its.mesh.emission) * weight_bsdf;
                        }
                    }
                }
                _ => {}
            }
        }
        l_i
    }
}
