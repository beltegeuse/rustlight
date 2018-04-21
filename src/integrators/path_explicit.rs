use integrators::*;
use paths::path::*;
use paths::vertex::*;
use scene::*;
use structure::*;

pub struct IntegratorUniPath {
    pub max_depth: Option<u32>,
}

impl Integrator<Color> for IntegratorUniPath {
    fn compute<S: Sampler>(&self, (ix, iy): (u32, u32), scene: &Scene, sampler: &mut S) -> Color {
        match Path::from_sensor((ix, iy), scene, sampler, self.max_depth) {
            None => Color::zero(),
            Some(path) => {
                let mut l_i = Color::zero();
                for (i, vertex) in path.vertices.iter().enumerate() {
                    match vertex {
                        &Vertex::Surface(ref v) => {
                            ///////////////////////////////
                            // Sample the light explicitly
                            let light_record = scene.sample_light(
                                &v.its.p,
                                sampler.next(),
                                sampler.next(),
                                sampler.next2d(),
                            );
                            let light_pdf = match light_record.pdf {
                                PDF::SolidAngle(v) => v,
                                _ => panic!("Unsupported light pdf type for pdf connection."),
                            };

                            let d_out_local = v.its.frame.to_local(light_record.d);
                            if light_record.is_valid() && scene.visible(&v.its.p, &light_record.p)
                                && d_out_local.z > 0.0
                            {
                                // Compute the contribution of direct lighting
                                if let PDF::SolidAngle(pdf_bsdf) =
                                    v.its.mesh.bsdf.pdf(&v.its.uv, &v.its.wi, &d_out_local)
                                {
                                    // Compute MIS weights
                                    let weight_light = mis_weight(light_pdf, pdf_bsdf);
                                    l_i += weight_light * v.throughput
                                        * v.its.mesh.bsdf.eval(&v.its.uv, &v.its.wi, &d_out_local)
                                        * light_record.weight;
                                }
                            }

                            /////////////////////////////////////////
                            // BSDF Sampling
                            // For the first hit, no MIS can be used
                            if i == 1 {
                                if v.its.cos_theta() > 0.0 {
                                    l_i += v.its.mesh.emission * v.throughput;
                                }
                            } else {
                                // We need to use MIS as we can generate this path
                                // using another technique
                                if v.its.mesh.is_light() && v.its.cos_theta() > 0.0 {
                                    let (pred_vertex_pos, pred_vertex_pdf) =
                                        match &path.vertices[i - 1] {
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
                                                dir: path.edges[i - 1].d,
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
    }
}
