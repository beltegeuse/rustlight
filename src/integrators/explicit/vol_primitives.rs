use crate::accel::*;
use crate::integrators::*;
use crate::paths::path::*;
use crate::paths::vertex::*;
use crate::samplers;
use crate::structure::AABB;
use crate::volume::*;
use cgmath::{EuclideanSpace, InnerSpace, Point2, Point3, Vector3};

pub struct IntegratorVolPrimitives {
    pub nb_primitive: usize,
    pub max_depth: Option<u32>,
    pub beams: bool,
}

pub struct TechniqueVolPrimitives {
    pub max_depth: Option<u32>,
    pub samplings: Vec<Box<dyn SamplingStrategy>>,
    pub flux: Option<Color>,
}

// TODO: Need to refactor this
impl Technique for TechniqueVolPrimitives {
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

// -------- Point representation
struct Photon {
    pos: Point3<f32>,
    d_in: Vector3<f32>,
    phase_function: PhaseFunction,
    radiance: Color,
    radius: f32,
}
impl BVHElement<f32> for Photon {
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

// ------------ Beam representation
// TODO: Short beam representation
// Note that this same structure will be used
// to split the photon beams, to accelerate the 
// BVH intersection.
struct PhotonBeam {
    o: Point3<f32>,
    d: Vector3<f32>,
    length: f32,
    phase_function: PhaseFunction,
    radiance: Color,
    radius: f32,
}
struct PhotonBeamIts {
    u: f32, // Kernel
    v: f32, // Beam
    w: f32, // Camera
    sin_theta: f32
}
impl BVHElement<PhotonBeamIts> for PhotonBeam {
    // Used to build AABB hierachy
    fn aabb(&self) -> AABB {
        let mut aabb = AABB::default();
        let radius = Point3::new(self.radius, self.radius, self.radius);
        aabb = aabb.union_vec(&(self.o - radius));
        aabb = aabb.union_vec(&(self.o.to_vec() + radius.to_vec()));
        aabb = aabb.union_vec(&(self.o + self.d * self.length - radius));
        aabb = aabb.union_vec(&(self.o + self.d * self.length + radius.to_vec()).to_vec());
        aabb
    }
    // Used to construct AABB (by sorting elements)
    fn position(&self) -> Point3<f32> {
        // Middle of the photon beam
        self.o + self.d * self.length * 0.5
    }

    // This is edge to edge intersection
    // This code is adapted from UPBP
    fn intersection(&self, r: &Ray) -> Option<PhotonBeamIts> {
        // const Vector d1d2c = cross(_ray.d, getDir());
        let d1d2c = r.d.cross(self.d);

        // Square of the sine between the two lines (||cross(d1, d2)|| = sinTheta).
        let sin_theta_sqr = d1d2c.magnitude2();
        
        // Lines too far apart.
        let ad = (self.o - r.o).dot(d1d2c);
        if ad * ad >= (self.radius * self.radius) * sin_theta_sqr {
            return None;
        }

        // Cosine between the two lines.
        let d1d2 = r.d.dot(self.d);
        let d1d2_sqr = d1d2 * d1d2;
        let d1d2_sqr_minus1 = d1d2_sqr - 1.0;

        // Parallel lines?
        if  d1d2_sqr_minus1 < 1e-5 && d1d2_sqr_minus1 > -1e-5 {
            return None;
        }
        let d1o1 = r.d.dot(r.o.to_vec());
        let d1o2 = r.d.dot(self.o.to_vec());
        
        // Out of range on ray 1.
        let w = (d1o1 - d1o2 - d1d2 * (self.d.dot(r.o.to_vec()) - self.d.dot(self.o.to_vec()))) / d1d2_sqr_minus1;
        if w <= r.tnear || w >= r.tfar {
            return None;
        }

    
        // Out of range on ray 2.
        let v = (w + d1o1 - d1o2) / d1d2;
        if v <= 0.0 || v >= self.length || !v.is_finite() {
            return None;
        }

        let sin_theta = sin_theta_sqr.sqrt();
        let u = ad.abs() / sin_theta;
        Some(PhotonBeamIts {
            u, v, w, sin_theta
        })
    }
}



impl TechniqueVolPrimitives {
    fn convert_beams<'scene>(
        &self,
        path: &Path<'scene, '_>,
        scene: &'scene Scene,
        vertex_id: VertexID,
        beams: &mut Vec<PhotonBeam>,
        radius: f32,
        flux: Color,
    ) {
        let m = scene.volume.as_ref().unwrap();
        match path.vertex(vertex_id) {
            Vertex::Surface(ref v) => {
                // Continue to bounce...
                for edge in &v.edge_out {
                    let edge = path.edge(*edge);
                    if let Some(vertex_next_id) = edge.vertices.1 {
                        beams.push(PhotonBeam {
                            o: v.its.p,
                            d: edge.d,
                            length: edge.dist.unwrap(),
                            phase_function: PhaseFunction::Isotropic(),
                            radiance: flux * edge.weight, 
                            radius,
                        });

                        self.convert_beams(
                            path,
                            scene,
                            vertex_next_id,
                            beams,
                            radius,
                            flux * edge.weight * edge.rr_weight,
                        );
                    }
                }
            }
            Vertex::Volume(ref v) => {
                // Continue to bounce...
                for edge in &v.edge_out {
                    let edge = path.edge(*edge);
                    if let Some(vertex_next_id) = edge.vertices.1 {
                        beams.push(PhotonBeam {
                            o: v.pos,
                            d: edge.d,
                            length: edge.dist.unwrap(),
                            phase_function: PhaseFunction::Isotropic(),
                            radiance: flux * edge.weight, 
                            radius,
                        });

                        self.convert_beams(
                            path,
                            scene,
                            vertex_next_id,
                            beams,
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
                        beams.push(PhotonBeam {
                            o: v.pos,
                            d: edge.d,
                            length: edge.dist.unwrap(),
                            phase_function: PhaseFunction::Isotropic(),
                            radiance: flux * edge.weight, 
                            radius,
                        });

                        self.convert_beams(
                            path,
                            scene,
                            next_vertex_id,
                            beams,
                            radius,
                            edge.weight * flux * edge.rr_weight,
                        );
                    }
                }
            }
            Vertex::Sensor(ref _v) => {}
        }
    }

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

impl Integrator for IntegratorVolPrimitives {
    fn compute(&mut self, accel: &dyn Acceleration, scene: &Scene) -> BufferCollection {
        if self.beams {
            info!("Using photon beams to render PM");
        }
        info!("Generating the light paths...");
        let buffernames = vec![String::from("primal")];
        let mut sampler = samplers::independent::IndependentSampler::default();
        let mut nb_path_shot = 0;
        let mut photons = vec![];
        let mut beams = vec![];

        let emitters = scene.emitters_sampler();
        while true {
            let samplings: Vec<Box<dyn SamplingStrategy>> =
                vec![Box::new(DirectionalSamplingStrategy { from_sensor: false })];
            let mut technique = TechniqueVolPrimitives {
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
            if self.beams {
                technique.convert_beams(&path, scene, root[0].0, &mut beams, 0.001, Color::one());
                if beams.len() >= self.nb_primitive as usize {
                    break;
                } 
            } else {
                technique.convert_photons(&path, scene, root[0].0, &mut photons, 0.001, Color::one());
                if photons.len() >= self.nb_primitive as usize {
                    break;
                } 
            }
            nb_path_shot += 1;
        }

        // Here we will cut the number of beams into subbeams
        // this will improve the performance of the BVH gathering
        // Note that for the moment it works only for short query.
        if self.beams {
            const SPLIT: usize = 5; // TODO: Make as parameter if sensitive
            let avg_split = beams.iter().map(|b| b.length).sum::<f32>() / (beams.len() * SPLIT) as f32;
            let nb_new_beams = beams.iter().map(|b| (b.length / avg_split).ceil() as u32).sum::<u32>();
            info!("Splitting beams: {} avg size | {} new beams", avg_split, nb_new_beams);

            let mut new_beams = vec![];
            new_beams.reserve(nb_new_beams as usize);
            for b in beams {
                let number_split = (b.length / avg_split).ceil() as u32; 
                let new_length = b.length / number_split as f32;
                for i in 0..number_split {
                    new_beams.push(PhotonBeam {
                        o: b.o + b.d*(i as f32)*new_length,
                        d: b.d,
                        length: new_length,
                        phase_function: b.phase_function.clone(), // FIXME: Might be wrong when deadling with spatially varying phase functions
                        radiance: b.radiance.clone(),
                        radius: b.radius,
                    })
                }
            }
            beams = new_beams;
        }

        // Build acceleration structure
        // Note that the radius here is fixed
        info!("Construct BVH...");
        let (bvh_photon, bvh_beams) = if self.beams {
            (None, Some(BHVAccel::create(beams)))
        } else {
            (Some(BHVAccel::create(photons)), None)
        };

        // Generate the image block to get VPL efficiently
        let mut image_blocks = generate_img_blocks(scene, &buffernames);

        // Render the image blocks VPL integration
        info!("Gathering Photons (BRE/Beams)...");
        let progress_bar = Mutex::new(ProgressBar::new(image_blocks.len() as u64));
        let norm_photon = 1.0 / nb_path_shot as f32;
        info!(" - Number of path generated: {}", nb_path_shot);
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

                            let m = scene.volume.as_ref().unwrap();
                            if self.beams {
                                let bvh = bvh_beams.as_ref().unwrap();
                                for(beam_its, b_id) in bvh.gather(ray) {
                                    let beam: &PhotonBeam = &bvh.elements[b_id];

                                    // Evaluate the transmittance
                                    let transmittance = {
                                        let mut ray_tr = Ray::new(ray.o, ray.d);
                                        ray_tr.tfar = beam_its.w;
                                        m.transmittance(ray_tr)
                                    };

                                    // Evaluate the phase function
                                    // Note that we need to add sigma_s here, as we create a new vertex
                                    let phase_func = m.sigma_s * beam.phase_function.eval(&(-ray.d), &(-beam.d));

                                    // Jacobian * Kernel
                                    let weight = (1.0 / beam_its.sin_theta) * (0.5 / beam.radius);
                                    c += beam.radiance
                                     * transmittance
                                     * phase_func
                                     * weight
                                     * norm_photon;
                                }
                            } else {
                                let bvh = bvh_photon.as_ref().unwrap();
                                for (dist, p_id) in bvh.gather(ray) {
                                    let photon: &Photon = &bvh.elements[p_id];
    
                                    // Evaluate the transmittance
                                    let transmittance = {
                                        let mut ray_tr = Ray::new(ray.o, ray.d);
                                        ray_tr.tfar = dist;
                                        m.transmittance(ray_tr)
                                    };
    
                                    // Evaluate the phase function
                                    // FIXME: Check the direction of d_in
                                    let phase_func =
                                        photon.phase_function.eval(&(-ray.d), &photon.d_in);
    
                                    // Kernel (2D in case of BRE)
                                    let weight = 1.0 / (std::f32::consts::PI * photon.radius.powi(2));
    
                                    // Sum all values
                                    c += photon.radiance
                                        * transmittance
                                        * phase_func
                                        * weight
                                        * norm_photon;
                                }
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
