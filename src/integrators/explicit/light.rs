use crate::integrators::*;
use crate::paths::strategies::*;
use crate::paths::{path::*, vertex::*};
use cgmath::InnerSpace;
use cgmath::Point2;

pub struct IntegratorLightTracing {
    pub max_depth: Option<u32>,
    pub min_depth: Option<u32>,
    pub render_surface: bool,
    pub render_volume: bool,
}

/// This structure is responsible to the graph generation
pub struct TechniqueLightTracing {
    pub max_depth: Option<u32>,
    pub min_depth: Option<u32>,
    pub samplings: Vec<Box<dyn SamplingStrategy>>, //< Might be not ideal...
    // To be able to select only a subset of the light transport
    pub render_surface: bool,
    pub render_volume: bool,
}

impl Technique for TechniqueLightTracing {
    fn expand(&self, _vertex: &Vertex, depth: u32) -> bool {
        self.max_depth.map_or(true, |max| depth < max)
    }

    fn strategies(&self, _vertex: &Vertex) -> &Vec<Box<dyn SamplingStrategy>> {
        &self.samplings
    }
}
impl TechniqueLightTracing {
    fn evaluate<'scene>(
        &self,
        depth: u32,
        path: &Path<'scene>,
        accel: &dyn Acceleration,
        scene: &'scene Scene,
        vertex_id: VertexID,
        bitmap: &mut BufferCollection,
        flux: Color,
    ) {
        let accumulate = match self.min_depth {
            Some(v) => v <= depth,
            None => true,
        };

        // Splat current vertex
        match path.vertex(vertex_id) {
            Vertex::Volume {
                pos,
                phase_function,
                d_in,
                ..
            } => {
                if accumulate && self.render_volume {
                    let pos_sensor = scene.camera.position();
                    let d = (pos_sensor - pos).normalize();
                    if accel.visible(pos, &pos_sensor) {
                        // Splat the contribution
                        if let Some((importance, uv)) = scene.camera.sample_direct(pos) {
                            let m = scene.volume.as_ref().unwrap();

                            // Compute BSDF for the splatting
                            let bsdf_value = phase_function.eval(d_in, &d);

                            // If medium, need to take into account the transmittance
                            let transmittance = {
                                let mut ray = Ray::new(*pos, d);
                                ray.tfar = (pos - pos_sensor).magnitude();
                                m.transmittance(ray)
                            };

                            // Accumulate the results
                            bitmap.accumulate_safe(
                                Point2::new(uv.x as i32, uv.y as i32),
                                flux * importance * bsdf_value * transmittance,
                                &"primal".to_owned(),
                            );
                        }
                    }
                }
            }
            Vertex::Surface { its, .. } => {
                if accumulate && self.render_surface {
                    // Chech the visibility from the point to the sensor
                    // we also exclude smooth BSDF as there is 0 chance to get
                    // a succesful connection
                    let pos_sensor = scene.camera.position();
                    let d = (pos_sensor - its.p).normalize();
                    if !its.mesh.bsdf.bsdf_type().is_smooth() && accel.visible(&its.p, &pos_sensor)
                    {
                        // Splat the contribution
                        if let Some((importance, uv)) = scene.camera.sample_direct(&its.p) {
                            // Compute BSDF for the splatting
                            let wo_local = its.frame.to_local(d);
                            let wi_global = its.frame.to_world(its.wi);
                            let bsdf_value = its.mesh.bsdf.eval(
                                &its.uv,
                                &its.wi,
                                &wo_local,
                                Domain::SolidAngle,
                                Transport::Radiance,
                            );
                            let correction =
                                (its.wi.z * d.dot(its.n_g)) / (wo_local.z * wi_global.dot(its.n_g));

                            // If medium, need to take into account the transmittance
                            let transmittance = if let Some(ref m) = scene.volume {
                                let mut ray = Ray::new(its.p, d);
                                ray.tfar = (its.p - pos_sensor).magnitude();
                                m.transmittance(ray)
                            } else {
                                Color::one()
                            };

                            // Accumulate the results
                            bitmap.accumulate_safe(
                                Point2::new(uv.x as i32, uv.y as i32),
                                flux * importance * bsdf_value * correction * transmittance,
                                &"primal".to_owned(),
                            );
                        }
                    }
                }
            }
            Vertex::Light {
                pos, n, edge_out, ..
            } => {
                if accumulate && self.render_surface {
                    // For now, we will use the edge to check the light
                    let edge_out = edge_out.unwrap();
                    let edge = path.edge(edge_out);
                    match edge.pdf_direction {
                        PDF::SolidAngle(_) => {
                            let pos_sensor = scene.camera.position();
                            let d = (pos_sensor - pos).normalize();
                            if accel.visible(pos, &pos_sensor) {
                                if let Some((importance, uv)) = scene.camera.sample_direct(pos) {
                                    let transmittance = if let Some(ref m) = scene.volume {
                                        let mut ray = Ray::new(*pos, d);
                                        ray.tfar = (pos - pos_sensor).magnitude();
                                        m.transmittance(ray)
                                    } else {
                                        Color::one()
                                    };

                                    bitmap.accumulate_safe(
                                        Point2::new(uv.x as i32, uv.y as i32),
                                        transmittance
                                            * flux
                                            * importance
                                            * d.dot(*n)
                                            * std::f32::consts::FRAC_1_PI,
                                        &"primal".to_owned(),
                                    );
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }

        // Go to the next vertex
        match path.vertex(vertex_id) {
            Vertex::Volume { edge_out, .. } | Vertex::Surface { edge_out, .. } => {
                for edge_id in edge_out {
                    let edge = path.edge(*edge_id);
                    if let Some(vertex_next) = edge.vertices.1 {
                        self.evaluate(
                            depth + 1,
                            path,
                            accel,
                            scene,
                            vertex_next,
                            bitmap,
                            flux * edge.weight * edge.rr_weight,
                        );
                    }
                }
            }
            Vertex::Light { edge_out, .. } => {
                if let Some(edge_id) = edge_out {
                    let edge = path.edge(*edge_id);
                    if let Some(next_vertex) = edge.vertices.1 {
                        self.evaluate(
                            depth + 1,
                            path,
                            accel,
                            scene,
                            next_vertex,
                            bitmap,
                            edge.weight * flux * edge.rr_weight,
                        );
                    }
                }
            }
            _ => {}
        }
    }
}

impl Integrator for IntegratorLightTracing {
    fn compute(
        &mut self,
        sampler: &mut dyn Sampler,
        accel: &dyn Acceleration,
        scene: &Scene,
    ) -> BufferCollection {
        // Number of samples that the system will trace
        // The strategy for multithread is to have 4 job per threads
        // All job will have the same number of samples to deal with
        let nb_threads = rayon::current_num_threads();
        let nb_jobs = nb_threads * 4;
        let mut samplers = (0..nb_jobs)
            .map(|_| sampler.clone_box())
            .collect::<Vec<_>>();

        // Ajust the number of light path that we need to generate
        let nb_samples = (scene.nb_samples
            * ((scene.camera.size().x * scene.camera.size().y) as usize))
            / nb_jobs as usize;

        // Global information
        let progress_bar = Mutex::new(ProgressBar::new(samplers.len() as u64));
        let buffer_names = vec![String::from("primal")];
        let img = Mutex::new(BufferCollection::new(
            Point2::new(0, 0),
            *scene.camera.size(),
            &buffer_names,
        ));

        let pool = generate_pool(scene);
        pool.install(|| {
            samplers.par_iter_mut().for_each(|s| {
                let mut my_img =
                    BufferCollection::new(Point2::new(0, 0), *scene.camera.size(), &buffer_names);

                // Initialize the strategy and the path
                // the path will be reused for each samples generated
                // The sampling strategies
                let samplings: Vec<Box<dyn SamplingStrategy>> = vec![Box::new(
                    crate::paths::strategies::directional::DirectionalSamplingStrategy {
                        transport: Transport::Importance,
                    },
                )];
                // Do the sampling here
                let mut technique = TechniqueLightTracing {
                    max_depth: self.max_depth,
                    min_depth: self.min_depth,
                    samplings,
                    render_surface: self.render_surface,
                    render_volume: self.render_volume,
                };
                let mut path = Path::default();

                (0..nb_samples).for_each(|_| {
                    path.clear();
                    let root = path.from_light(scene, s.as_mut());
                    generate(&mut path, root.0, accel, scene, s.as_mut(), &mut technique);
                    // Evaluate the path generated using camera splatting operation
                    technique.evaluate(0, &path, accel, scene, root.0, &mut my_img, root.1);
                });

                // Scale and add the results
                my_img.scale(1.0 / (nb_samples as f32));
                {
                    img.lock().unwrap().accumulate_bitmap(&my_img);
                    progress_bar.lock().unwrap().inc();
                }
            });
        });

        // All job are independent, so we just merge them...
        let mut img: BufferCollection = img.into_inner().unwrap();
        img.scale(1.0 / nb_jobs as f32);
        img.scale((scene.camera.img.x * scene.camera.img.y) as f32);
        img
    }
}
