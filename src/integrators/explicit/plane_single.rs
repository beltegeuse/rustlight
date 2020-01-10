use crate::accel::*;
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

// TODO: The class is very similar to photon planes
struct SinglePhotonPlane {
    o: Point3<f32>,
    d0: Vector3<f32>,
    d1: Vector3<f32>,
    length0: f32,
	length1: f32,
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

pub enum SinglePlaneStrategy {
	UV,
	VT,
	UT,
	Average,
	DiscreteMIS
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
		let emitters = scene.meshes.iter().filter(|m| !m.emission.is_zero()).collect::<Vec<_>>();
		if emitters.len() != 1 {
			panic!("Do not support multiple light source! The number of light source is: {}", emitters.len());
		}

		let emitter = emitters[0];
		info!("Emitter vertices: {:?}", emitter.vertices);
		info!("Emitter indices: {:?}", emitter.indices);
		if emitter.vertices.len() != 3 && emitter.indices.len() != 2 {
			panic!("Only support rectangular emitters");
		}
		let o_light = Point3::from_vec(emitter.vertices[0]);
		let u_light = emitter.vertices[1] - emitter.vertices[0];
		let v_light = emitter.vertices[3] - emitter.vertices[0];


		info!("o : {:?}", o_light);
		info!("u : {:?} ({})", u_light, u_light.magnitude());
		info!("v : {:?} ({})", v_light, v_light.magnitude());

		let n = u_light.cross(v_light).normalize();
		info!("Light source normal: {:?}", n);

		// Create the planes
		let m = scene.volume.as_ref().unwrap();
		let mut sampler = samplers::independent::IndependentSampler::default();
		let mut planes = vec![];
		let mut number_plane_gen = 0;
		while number_plane_gen < self.nb_primitive {
			// Generate position on the rectangular light source
			let sample = sampler.next2d();
			let o = o_light + u_light * sample.x + v_light* sample.y;
			let d = {
				let d_out = cosine_sample_hemisphere(sampler.next2d());
                if d_out.z == 0.0 {
					// Start to generate a new plane again
                    continue; 
                }
				let frame = Frame::new(n);
				frame.to_world(d_out)
			};
			number_plane_gen += 1;

			// Sample a distance, we create a primitive even if we hit a surface 
			// (as it might be similar to photon plane)
			let ray_med = Ray::new(o, d);
			// TODO: Check if it is the code
			// Need to check the intersection distance iff need to failed ...
            // ray_med.tfar = intersection_distance;
			let mrec = m.sample(&ray_med, sampler.next2d());
			
			// Let's create all type of planes...
			{
				let o_plane = o_light + d * mrec.continued_t;
				let l1 = u_light.magnitude();
				let l2 = v_light.magnitude();
				planes.push(SinglePhotonPlane {
					o: o_plane, 
					d0: u_light / l1,
					d1: v_light / l2,
					length0: l1,
					length1: l2,
					sample,
					plane_type: PlaneType::UV,
				});
			}

			// VT
			{
				let o_plane = o_light + u_light * sample.x;
				let l1 = v_light.magnitude();
				let l2 = mrec.continued_t;
				planes.push(SinglePhotonPlane {
					o: o_plane, 
					d0: v_light / l1,
					d1: d,
					length0: l1,
					length1: l2,
					sample,
					plane_type: PlaneType::VT,
				});
			}

			// UT
			{
				let o_plane = o_light + v_light * sample.y;
				let l1 = u_light.magnitude();
				let l2 = mrec.continued_t;
				planes.push(SinglePhotonPlane {
					o: o_plane, 
					d0: u_light / l1,
					d1: d,
					length0: l1,
					length1: l2,
					sample,
					plane_type: PlaneType::UT,
				});
			}
		}

		// Build the BVH to speedup the computation...
		let bvh_plane = BHVAccel::create(planes);
		
		// Generate the image block to get VPL efficiently
		let buffernames = vec![String::from("primal")];
		let mut image_blocks = generate_img_blocks(scene, &buffernames);

		// This is a hack to get the light source flux
		let total_flux = {
			let emitter_sampler = scene.emitters_sampler();
			let mut sum = Color::zero();
			for e in emitter_sampler.emitters {
				sum += e.flux()
			}
			sum
		};
		
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
							// for plane in &planes {
								// let plane_its = plane.intersection(&ray);
								// if plane_its.is_none() {
								// 	continue;
								// }
								// let plane_its = plane_its.unwrap();

								let p_hit = ray.o + ray.d * plane_its.t_cam;
								let p_light = match plane.plane_type {
									PlaneType::UV => o_light + u_light * plane_its.t0 + v_light* plane_its.t1,
									PlaneType::VT => o_light + u_light * plane.sample.x + v_light* plane_its.t0,
									PlaneType::UT => o_light + v_light * plane.sample.y + u_light * plane_its.t0,
									_ => unimplemented!(),
								};
								let do_integration = match self.strategy {
									SinglePlaneStrategy::UT => plane.plane_type == PlaneType::UT,
									SinglePlaneStrategy::VT => plane.plane_type == PlaneType::VT,
									SinglePlaneStrategy::UV => plane.plane_type == PlaneType::UV,
									SinglePlaneStrategy::Average => plane.plane_type != PlaneType::UAlphaT,
									SinglePlaneStrategy::DiscreteMIS => plane.plane_type != PlaneType::UAlphaT,
								};

								if !do_integration {
									continue;
								}

								if accel.visible(&p_hit, &	p_light) {
							
									let jacobian = plane.d1.cross(plane.d0).dot(ray.d).abs();
									if jacobian != 0.0 {
										let transmittance = {
											let mut ray_tr = Ray::new(ray.o, ray.d);
											ray_tr.tfar = plane_its.t_cam;
											m.transmittance(ray_tr)
										};
										let rho = phase_function.eval(&(-ray.d), &(p_light - p_hit).normalize());
										let w = match self.strategy {
											SinglePlaneStrategy::UT |
											SinglePlaneStrategy::UV |
											SinglePlaneStrategy::VT => 1.0,
											SinglePlaneStrategy::Average => 1.0 / 3.0,
											SinglePlaneStrategy::DiscreteMIS => unimplemented!(),
										};

										c += w * rho * transmittance * m.sigma_s * (total_flux / jacobian) * ( 1.0 / self.nb_primitive as f32);
									}
									
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

