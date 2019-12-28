use crate::accel::*;
use crate::integrators::*;
use crate::paths::path::*;
use crate::paths::vertex::*;
use crate::samplers;
use crate::structure::AABB;
use crate::volume::*;
use cgmath::{EuclideanSpace, InnerSpace, Point2, Point3, Vector3};

pub struct IntegratorBRE {
    pub nb_photons: usize,
    pub max_depth: Option<u32>,
}

pub struct TechniqueBRE {
    pub max_depth: Option<u32>,
    pub samplings: Vec<Box<dyn SamplingStrategy>>,
    pub flux: Option<Color>,
}

// TODO: Need to refactor this
impl Technique for TechniqueBRE {
    fn init<'scene, 'emitter>(
        &mut self,
        path: &mut Path<'scene, 'emitter>,
        _accel: &dyn Acceleration,
        _scene: &'scene Scene,
        sampler: &mut dyn Sampler,
        emitters: &'emitter EmitterSampler,
    ) -> Vec<(VertexID, Color)> {
        let (emitter, sampled_point, flux) = emitters.random_sample_emitter_position(
            sampler.next(),
            sampler.next(),
            sampler.next2d(),
        );
        let emitter_vertex = Vertex::Light(EmitterVertex {
            pos: sampled_point.p,
            n: sampled_point.n,
            emitter,
            edge_in: None,
            edge_out: None,
        });
        self.flux = Some(flux); // Capture the scaled flux
        vec![(path.register_vertex(emitter_vertex), Color::one())]
    }

    fn expand(&self, _vertex: &Vertex, depth: u32) -> bool {
        self.max_depth.map_or(true, |max| depth < max)
    }

    fn strategies(&self, _vertex: &Vertex) -> &Vec<Box<dyn SamplingStrategy>> {
        &self.samplings
    }
}

struct Photon {
    pos: Point3<f32>,
    d_in: Vector3<f32>,
    phase_function: PhaseFunction,
    radiance: Color,
    radius: f32,
}

impl BVHElement for Photon {
    // Used to build AABB hierachy
    fn aabb(&self) -> AABB {
        let mut aabb = AABB::default();
        let radius = Point3::new(self.radius, self.radius, self.radius);
        aabb = aabb.union_vec(&(self.pos - radius));
        aabb = aabb.union_vec(&(self.pos.to_vec() + radius.to_vec()));
        aabb
    }
    // Used to construct AABB (by sorting elements)
    fn position(&self) -> Point3<f32> {
        self.pos
    }

    // Used when collecting the different objects
    fn intersection(&self, r: &Ray) -> Option<f32> {
        let d_p = self.pos - r.o;
        let dot = d_p.dot(r.d);
        if dot <= 0.0 || dot > r.tfar {
            // No intersection
            None
        } else {
            let p = r.o + r.d * dot;
            let dist = (self.pos - p).magnitude2();
            if dist > self.radius * self.radius {
                None
            } else {
                Some(dot)
            }
        }
    }
}

impl TechniqueBRE {
    fn convert_photons<'scene>(
        &self,
        path: &Path<'scene, '_>,
        scene: &'scene Scene,
        vertex_id: VertexID,
        photons: &mut Vec<Photon>,
        radius: f32,
        flux: Color,
    ) {
        match path.vertex(vertex_id) {
            Vertex::Surface(ref v) => {
                // Continue to bounce...
                for edge in &v.edge_out {
                    let edge = path.edge(*edge);
                    if let Some(vertex_next_id) = edge.vertices.1 {
                        self.convert_photons(
                            path,
                            scene,
                            vertex_next_id,
                            photons,
                            radius,
                            flux * edge.weight * edge.rr_weight,
                        );
                    }
                }
            }
            Vertex::Volume(ref v) => {
                photons.push(Photon {
                    pos: v.pos,
                    d_in: v.d_in,
                    phase_function: v.phase_function.clone(),
                    radiance: flux,
                    radius,
                });

                // Continue to bounce...
                for edge in &v.edge_out {
                    let edge = path.edge(*edge);
                    if let Some(vertex_next_id) = edge.vertices.1 {
                        self.convert_photons(
                            path,
                            scene,
                            vertex_next_id,
                            photons,
                            radius,
                            flux * edge.weight * edge.rr_weight,
                        );
                    }
                }
            }
            Vertex::Light(ref v) => {
                let flux = *self.flux.as_ref().unwrap();
                if let Some(edge) = v.edge_out {
                    let edge = path.edge(edge);
                    if let Some(next_vertex_id) = edge.vertices.1 {
                        self.convert_photons(
                            path,
                            scene,
                            next_vertex_id,
                            photons,
                            radius,
                            edge.weight * flux * edge.rr_weight,
                        );
                    }
                }
            }
            Vertex::Sensor(ref _v) => {}
        }
    }
}

impl Integrator for IntegratorBRE {
    fn compute(&mut self, accel: &dyn Acceleration, scene: &Scene) -> BufferCollection {
        info!("Generating the VPL...");
        let buffernames = vec![String::from("primal")];
        let mut sampler = samplers::independent::IndependentSampler::default();
        let mut nb_path_shot = 0;
        let mut photons = vec![];
        let emitters = scene.emitters_sampler();
        while photons.len() < self.nb_photons as usize {
            let samplings: Vec<Box<dyn SamplingStrategy>> =
                vec![Box::new(DirectionalSamplingStrategy { from_sensor: false })];
            let mut technique = TechniqueBRE {
                max_depth: self.max_depth,
                samplings,
                flux: None,
            };
            let mut path = Path::default();
            let root = generate(
                &mut path,
                accel,
                scene,
                &emitters,
                &mut sampler,
                &mut technique,
            );
            technique.convert_photons(&path, scene, root[0].0, &mut photons, 0.001, Color::one());
            nb_path_shot += 1;
        }

        // Build acceleration structure
        // Note that the radius here is fixed
        info!("Construct BVH...");
        let bvh = BHVAccel::create(photons);

        // Generate the image block to get VPL efficiently
        let mut image_blocks = generate_img_blocks(scene, &buffernames);

        // Render the image blocks VPL integration
        info!("Gathering Photons (BRE)...");
        let progress_bar = Mutex::new(ProgressBar::new(image_blocks.len() as u64));
        let norm_photon = 1.0 / nb_path_shot as f32;
        let pool = generate_pool(scene);
        pool.install(|| {
            image_blocks.par_iter_mut().for_each(|im_block| {
                let mut sampler = independent::IndependentSampler::default();
                for ix in 0..im_block.size.x {
                    for iy in 0..im_block.size.y {
                        for _ in 0..scene.nb_samples {
                            let (ix_c, iy_c) = (ix + im_block.pos.x, iy + im_block.pos.y);
                            let pix = Point2::new(
                                ix_c as f32 + sampler.next(),
                                iy_c as f32 + sampler.next(),
                            );
                            let mut ray = scene.camera.generate(pix);

                            // Get the max distance
                            let max_dist = match accel.trace(&ray) {
                                Some(x) => x.dist,
                                None => std::f32::MAX,
                            };
                            ray.tfar = max_dist;

                            // Get all photons intersected....
                            let mut c = Color::value(0.0);
                            let photons_its = bvh.gather(ray);
                            // dbg!(&photons_its);
                            let m = scene.volume.as_ref().unwrap();
                            for (dist, p_id) in photons_its {
                                let photon: &Photon = &bvh.elements[p_id];

                                // Evaluate the transmittance
                                let transmittance = {
                                    let mut ray_tr = Ray::new(ray.o, ray.d);
                                    ray_tr.tfar = dist;
                                    m.transmittance(ray_tr)
                                };

                                // Evaluate the phase function
                                let phase_func =
                                    photon.phase_function.eval(&(-ray.d), &photon.d_in);

                                // Kernel (2D in case of BRE)
                                let weight = 1.0 / (std::f32::consts::PI * photon.radius.powi(2));

                                // Sum all values
                                c += bvh.elements[p_id].radiance
                                    * transmittance
                                    * phase_func
                                    * weight
                                    * norm_photon;
                            }

                            im_block.accumulate(Point2 { x: ix, y: iy }, c, &"primal".to_owned());
                        }
                    }
                }
                im_block.scale(1.0 / (scene.nb_samples as f32));
                {
                    progress_bar.lock().unwrap().inc();
                }
            });
        });

        // Fill the image
        let mut image =
            BufferCollection::new(Point2::new(0, 0), *scene.camera.size(), &buffernames);
        for im_block in &image_blocks {
            image.accumulate_bitmap(im_block);
        }
        image
    }
}
