use crate::accel::*;
use crate::geometry::Mesh;
use crate::integrators::*;
use crate::math::*;
use crate::samplers;
use crate::structure::AABB;
use crate::volume::*;
use cgmath::{EuclideanSpace, InnerSpace, Point2, Point3, Vector3};

#[derive(PartialEq)]
pub enum PlaneType {
    UV,
    VT,
    UT,
    UAlphaT,
}

// Helper on the light source
struct RectangularLightSource {
    o: Point3<f32>,
    n: Vector3<f32>,
    u: Vector3<f32>,
    v: Vector3<f32>,
    u_l: f32,
    v_l: f32,
}
impl RectangularLightSource {
    fn from_shape(emitter: &Mesh) -> Self {
        info!("Emitter vertices: {:?}", emitter.vertices);
        info!("Emitter indices: {:?}", emitter.indices);
        if emitter.vertices.len() != 3 && emitter.indices.len() != 2 {
            panic!("Only support rectangular emitters");
        }
        let o = Point3::from_vec(emitter.vertices[0]);
        let u = emitter.vertices[1] - emitter.vertices[0];
        let v = emitter.vertices[3] - emitter.vertices[0];
        let u_l = u.magnitude();
        let v_l = v.magnitude();

        // Normalize vectors
        let u = u / u_l;
        let v = v / v_l;

        info!("o : {:?}", o);
        info!("u : {:?} ({})", u, u_l);
        info!("v : {:?} ({})", v, v_l);

        let n = u.cross(v);
        info!("Light source normal: {:?}", n);
        RectangularLightSource {
            o,
            n,
            u,
            v,
            u_l,
            v_l,
        }
    }
}

// TODO: The class is very similar to photon planes
struct SinglePhotonPlane {
    o: Point3<f32>,
    d0: Vector3<f32>,
    d1: Vector3<f32>,
    length0: f32,
    length1: f32,
    inv_pdf: f32,
    sample: Point2<f32>,   // Random number used to generate the plane
    plane_type: PlaneType, // How the plane have been generated
}

#[derive(Debug)]
struct PhotonPlaneIts {
    t_cam: f32,
    t0: f32,
    t1: f32,
    inv_det: f32,
}
impl BVHElement<PhotonPlaneIts> for SinglePhotonPlane {
    fn aabb(&self) -> AABB {
        let p0 = self.o + self.d0 * self.length0;
        let p1 = self.o + self.d1 * self.length1;
        let p2 = p0 + self.d1 * self.length1;
        let mut aabb = AABB::default();
        aabb = aabb.union_vec(&self.o.to_vec());
        aabb = aabb.union_vec(&p0.to_vec());
        aabb = aabb.union_vec(&p1.to_vec());
        aabb = aabb.union_vec(&p2.to_vec());
        aabb
    }
    // Used to construct AABB (by sorting elements)
    fn position(&self) -> Point3<f32> {
        // Middle of the photon plane
        // Note that it might be not ideal....
        self.o + self.d0 * self.length0 * 0.5 + self.d1 * self.length1 * 0.5
    }
    // This code is very similar to triangle intersection
    // except that we loose one test to make posible to
    // intersect planar primitives
    fn intersection(&self, r: &Ray) -> Option<PhotonPlaneIts> {
        let e0 = self.d0 * self.length0;
        let e1 = self.d1 * self.length1;

        let p = r.d.cross(e1);
        let det = e0.dot(p);
        if det.abs() < 1e-5 {
            return None;
        }

        let inv_det = 1.0 / det;
        let t = r.o - self.o;
        let t0 = t.dot(p) * inv_det;
        if t0 < 0.0 || t0 > 1.0 {
            return None;
        }

        let q = t.cross(e0);
        let t1 = r.d.dot(q) * inv_det;
        if t1 < 0.0 || t1 > 1.0 {
            return None;
        }

        let t_cam = e1.dot(q) * inv_det;
        if t_cam <= r.tnear || t_cam >= r.tfar {
            return None;
        }

        // Scale to the correct distance
        // In order to use correctly transmittance sampling
        let t1 = t1 * self.length1;
        let t0 = t0 * self.length0;

        Some(PhotonPlaneIts {
            t_cam,
            t0,
            t1,
            inv_det,
        })
    }
}
impl SinglePhotonPlane {
    fn light_position(
        &self,
        light: &RectangularLightSource,
        plane_its: &PhotonPlaneIts,
    ) -> Point3<f32> {
        match self.plane_type {
            PlaneType::UV => {
                light.o + light.u * light.u_l * plane_its.t0 + light.v * light.v_l * plane_its.t1
            }
            PlaneType::VT => {
                light.o + light.u * light.u_l * self.sample.x + light.v * light.v_l * plane_its.t0
            }
            PlaneType::UT => {
                light.o + light.v * light.v_l * self.sample.y + light.u * light.u_l * plane_its.t0
            }
            _ => unimplemented!(),
        }
    }
    fn contrib(&self, d: &Vector3<f32>) -> f32 {
        let jacobian = self.d1.cross(self.d0).dot(*d).abs();
        self.inv_pdf / jacobian
    }
    fn new(
        t: PlaneType,
        light: &RectangularLightSource,
        d: Vector3<f32>,
        sample: Point2<f32>,
        t_sampled: f32,
    ) -> Self {
        match t {
            PlaneType::UV => {
                let o_plane = light.o + d * t_sampled;
                SinglePhotonPlane {
                    o: o_plane,
                    d0: light.u,
                    d1: light.v,
                    // TODO: Why 2.0 in this case???
                    // Maybe it becase we can gather from the two side
                    inv_pdf: 2.0 / (light.u_l * light.v_l),
                    length0: light.u_l,
                    length1: light.v_l,
                    sample,
                    plane_type: PlaneType::UV,
                }
            }
            PlaneType::VT => {
                let o_plane = light.o + light.u * light.u_l * sample.x;
                SinglePhotonPlane {
                    o: o_plane,
                    d0: light.v,
                    d1: d,
                    inv_pdf: 1.0 / light.v_l,
                    length0: light.v_l,
                    length1: t_sampled,
                    sample,
                    plane_type: PlaneType::VT,
                }
            }
            PlaneType::UT => {
                let o_plane = light.o + light.v * light.v_l * sample.y;
                SinglePhotonPlane {
                    o: o_plane,
                    d0: light.u,
                    d1: d,
                    inv_pdf: 1.0 / light.u_l,
                    length0: light.u_l,
                    length1: t_sampled,
                    sample,
                    plane_type: PlaneType::UT,
                }
            }
            _ => unimplemented!(),
        }
    }
}

pub enum SinglePlaneStrategy {
    UV,
    VT,
    UT,
    Average,
    DiscreteMIS,
}

pub struct IntegratorSinglePlane {
    pub nb_primitive: usize,
    pub strategy: SinglePlaneStrategy,
}

impl Integrator for IntegratorSinglePlane {
    fn compute(&mut self, accel: &dyn Acceleration, scene: &Scene) -> BufferCollection {
        if scene.volume.is_none() {
            panic!("Volume integrator need a volume (add -m )");
        }
        // Extract the light source
        let emitters = scene
            .meshes
            .iter()
            .filter(|m| !m.emission.is_zero())
            .collect::<Vec<_>>();
        if emitters.len() != 1 {
            panic!(
                "Do not support multiple light source! The number of light source is: {}",
                emitters.len()
            );
        }

        let rect_light = {
            let emitter = emitters[0];
            info!("Emitter vertices: {:?}", emitter.vertices);
            info!("Emitter indices: {:?}", emitter.indices);
            if emitter.vertices.len() != 3 && emitter.indices.len() != 2 {
                panic!("Only support rectangular emitters");
            }
            RectangularLightSource::from_shape(emitter)
        };

        // TODO: Simplify the code...
        let total_flux = {
            let emitter_sampler = scene.emitters_sampler();
            let mut sum = Color::zero();
            for e in emitter_sampler.emitters {
                sum += e.flux()
            }
            sum
        };

        let generate_plane = |t: PlaneType,
                              light: &RectangularLightSource,
                              sampler: &mut dyn Sampler,
                              m: &HomogenousVolume|
         -> SinglePhotonPlane {
            let d = {
                let mut d_out = cosine_sample_hemisphere(sampler.next2d());
                while d_out.z == 0.0 {
                    // Start to generate a new plane again
                    d_out = cosine_sample_hemisphere(sampler.next2d());
                }
                let frame = Frame::new(light.n);
                frame.to_world(d_out)
            };

            // FIXME: Faking position as it is not important for sampling the transmittance
            let ray_med = Ray::new(light.o, d);
            // TODO: Check if it is the code
            // Need to check the intersection distance iff need to failed ...
            // ray_med.tfar = intersection_distance;
            let mrec = m.sample(&ray_med, sampler.next2d());

            // Sample planes
            let sample = sampler.next2d();
            SinglePhotonPlane::new(t, light, d, sample, mrec.continued_t)
        };

        // Create the planes
        let m = scene.volume.as_ref().unwrap();
        let mut sampler = samplers::independent::IndependentSampler::default();
        let mut planes = vec![];
        let mut number_plane_gen = 0;
        while planes.len() < self.nb_primitive {
            match self.strategy {
                SinglePlaneStrategy::UT => {
                    planes.push(generate_plane(PlaneType::UT, &rect_light, &mut sampler, m))
                }
                SinglePlaneStrategy::VT => {
                    planes.push(generate_plane(PlaneType::VT, &rect_light, &mut sampler, m))
                }
                SinglePlaneStrategy::UV => {
                    planes.push(generate_plane(PlaneType::UV, &rect_light, &mut sampler, m))
                }
                SinglePlaneStrategy::DiscreteMIS | SinglePlaneStrategy::Average => {
                    // Generate 3 planes
                    planes.push(generate_plane(PlaneType::UV, &rect_light, &mut sampler, m));
                    planes.push(generate_plane(PlaneType::VT, &rect_light, &mut sampler, m));
                    planes.push(generate_plane(PlaneType::UT, &rect_light, &mut sampler, m));
                }
            }

            number_plane_gen += 1;
        }

        // Build the BVH to speedup the computation...
        let bvh_plane = BHVAccel::create(planes);

        // Generate the image block to get VPL efficiently
        let buffernames = vec![String::from("primal")];
        let mut image_blocks = generate_img_blocks(scene, &buffernames);

        // Gathering all planes
        info!("Gathering Single planes...");
        let progress_bar = Mutex::new(ProgressBar::new(image_blocks.len() as u64));
        let pool = generate_pool(scene);
        let phase_function = PhaseFunction::Isotropic();
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

                            // Now gather all planes
                            let mut c = Color::value(0.0);
                            for (plane_its, b_id) in bvh_plane.gather(ray) {
                                let plane = &bvh_plane.elements[b_id];
                                // This code is if we do not use BVH
                                // for plane in &planes {
                                // let plane_its = plane.intersection(&ray);
                                // if plane_its.is_none() {
                                // 	continue;
                                // }
                                // let plane_its = plane_its.unwrap();

                                let p_hit = ray.o + ray.d * plane_its.t_cam;
                                let p_light = plane.light_position(&rect_light, &plane_its);
                                if accel.visible(&p_hit, &p_light) {
                                    let transmittance = {
                                        let mut ray_tr = Ray::new(ray.o, ray.d);
                                        ray_tr.tfar = plane_its.t_cam;
                                        m.transmittance(ray_tr)
                                    };
                                    let rho = phase_function
                                        .eval(&(-ray.d), &(p_light - p_hit).normalize());
                                    let w = match self.strategy {
                                        SinglePlaneStrategy::UT
                                        | SinglePlaneStrategy::UV
                                        | SinglePlaneStrategy::VT => 1.0,
                                        SinglePlaneStrategy::Average => 1.0 / 3.0,
                                        SinglePlaneStrategy::DiscreteMIS => {
                                            // Need to compute all possible shapes
                                            let d = p_hit - p_light;
                                            // TODO: Not used
                                            let t_sampled = d.magnitude();
                                            let d = d / t_sampled;
                                            let planes = [
                                                SinglePhotonPlane::new(
                                                    PlaneType::UV,
                                                    &rect_light,
                                                    d,
                                                    plane.sample,
                                                    t_sampled,
                                                ),
                                                SinglePhotonPlane::new(
                                                    PlaneType::UT,
                                                    &rect_light,
                                                    d,
                                                    plane.sample,
                                                    t_sampled,
                                                ),
                                                SinglePhotonPlane::new(
                                                    PlaneType::VT,
                                                    &rect_light,
                                                    d,
                                                    plane.sample,
                                                    t_sampled,
                                                ),
											];
											// FIXME: Normally this code is unecessary
											// 	As we can reuse the plane retrived.
											// 	However, it seems to have a miss match between photon planes 
											//	contribution calculation.
											let debug_id = match plane.plane_type {
												PlaneType::UV => 0,
												PlaneType::UT => 1,
												PlaneType::VT => 2,
												_ => unimplemented!(),
											};

                                            planes[debug_id].contrib(&ray.d).powi(-1)
                                                / planes
                                                    .iter()
                                                    .map(|p| p.contrib(&ray.d).powi(-1))
                                                    .sum::<f32>()
                                        }
									};
									assert!(w > 0.0 && w < 1.0);

                                    // UV planes are not importance sampled (position/direction)
                                    // For other primitive, there such importance sampled approach.
                                    let flux = total_flux * plane.contrib(&ray.d);
                                    c += w
                                        * rho
                                        * transmittance
                                        * m.sigma_s
                                        * flux
                                        * (1.0 / number_plane_gen as f32);
                                }
                            }

                            im_block.accumulate(Point2 { x: ix, y: iy }, c, &"primal".to_owned());
                        }
                    }
                } // Image block
                im_block.scale(1.0 / (scene.nb_samples as f32));
                {
                    progress_bar.lock().unwrap().inc();
                }
            });
        });

        let mut image =
            BufferCollection::new(Point2::new(0, 0), *scene.camera.size(), &buffernames);
        for im_block in &image_blocks {
            image.accumulate_bitmap(im_block);
        }
        image
    }
}
