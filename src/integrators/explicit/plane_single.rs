use crate::accel::*;
use crate::geometry::Mesh;
use crate::integrators::*;
use crate::math::*;
use crate::structure::AABB;
use crate::volume::*;
use cgmath::{ElementWise, EuclideanSpace, InnerSpace, Point2, Point3, Vector3};

fn clamp<T: PartialOrd>(v: T, min: T, max: T) -> T {
    if v < min {
        min
    } else if v > max {
        max
    } else {
        v
    }
}

#[derive(PartialEq, Clone, Debug)]
pub enum PlaneType {
    UV,
    VT,
    UT,
    UAlphaT,
}

// Helper on the light source
pub struct RectangularLightSource {
    pub o: Point3<f32>,
    pub n: Vector3<f32>,
    pub u: Vector3<f32>,
    pub v: Vector3<f32>,
    pub u_l: f32,
    pub v_l: f32,
    pub emission: Color,
}
impl RectangularLightSource {
    pub fn from_shape(emitter: &Mesh) -> Self {
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

        let emission = match emitter.emission {
            crate::geometry::EmissionType::Color { v } => v,
            _ => panic!(),
        };

        RectangularLightSource {
            o,
            n,
            u,
            v,
            u_l,
            v_l,
            emission,
        }
    }
}

// TODO: The class is very similar to photon planes
#[derive(Clone, Debug)]
pub struct SinglePhotonPlane {
    o: Point3<f32>,            // Plane origin
    pub d0: Vector3<f32>,      // First edge (normalized)
    pub d1: Vector3<f32>,      // Second edge (normalized)
    length0: f32,              // First edge length
    length1: f32,              // Second edge length
    pub sample: Point2<f32>,   // Random number used to generate the plane (TODO: Unused?)
    pub plane_type: PlaneType, // How the plane have been generated
    pub weight: Color,         // This factor will vary between the different light sources
    id_emitter: usize,
    pub sample_alpha: f32,
}

#[derive(Debug)]
pub struct PhotonPlaneIts {
    pub t_cam: f32,
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
    pub fn light_position(
        &self,
        light: &RectangularLightSource,
        plane_its: &PhotonPlaneIts,
    ) -> Point3<f32> {
        match self.plane_type {
            PlaneType::UV => light.o + light.u * plane_its.t0 + light.v * plane_its.t1,
            PlaneType::VT | PlaneType::UT | PlaneType::UAlphaT => self.o + self.d0 * plane_its.t0,
        }
    }
    pub fn contrib(&self, d: &Vector3<f32>) -> Color {
        let jacobian = self.d1.cross(self.d0).dot(*d).abs();
        self.weight / jacobian
    }
    pub fn new(
        t: PlaneType,
        light: &RectangularLightSource,
        d: Vector3<f32>,
        sample: Point2<f32>,
        sample_alpha: f32,
        t_sampled: f32,
        id_emitter: usize,
        sigma_s: Color,
    ) -> Self {
        match t {
            PlaneType::UV => {
                let o_plane = light.o + d * t_sampled;
                SinglePhotonPlane {
                    o: o_plane,
                    d0: light.u,
                    d1: light.v,
                    length0: light.u_l,
                    length1: light.v_l,
                    sample,
                    plane_type: PlaneType::UV,
                    // Sigma_s is used to cancel out (as we sample a distance in planes)
                    weight: std::f32::consts::PI * light.emission / sigma_s,
                    id_emitter,
                    sample_alpha,
                }
            }
            PlaneType::VT => {
                let o_plane = light.o + light.u * light.u_l * sample.x;
                SinglePhotonPlane {
                    o: o_plane,
                    d0: light.v,
                    d1: d,
                    length0: light.v_l,
                    length1: t_sampled,
                    sample,
                    plane_type: PlaneType::VT,
                    // f() / (p(w_l) * p(u))
                    weight: std::f32::consts::PI * light.u_l * light.emission,
                    id_emitter,
                    sample_alpha,
                }
            }
            PlaneType::UT => {
                let o_plane = light.o + light.v * light.v_l * sample.y;
                SinglePhotonPlane {
                    o: o_plane,
                    d0: light.u,
                    d1: d,
                    length0: light.u_l,
                    length1: t_sampled,
                    sample,
                    plane_type: PlaneType::UT,
                    // f() / (p(w_l) * p(u))
                    weight: std::f32::consts::PI * light.v_l * light.emission,
                    id_emitter,
                    sample_alpha,
                }
            }
            PlaneType::UAlphaT => {
                let alpha = std::f32::consts::PI * sample_alpha;
                let o_plane = Point2::new(sample.x * light.u_l, sample.y * light.v_l);
                let d_plane: Vector2<f32> = Vector2::new(alpha.cos(), alpha.sin());

                // 2D AABB
                let plane2d_its = |d: Vector2<f32>, o: Point2<f32>| {
                    let t_0 = (-o.to_vec()).div_element_wise(d);
                    let t_1 = (Vector2::new(light.u_l, light.v_l) - o.to_vec()).div_element_wise(d);
                    let t_max_coord = Vector2::new(t_0.x.max(t_1.x), t_0.y.max(t_1.y));
                    o_plane + d * t_max_coord.x.min(t_max_coord.y)
                };

                // These are the intersection points in 2D (local coordinate)
                let p1_2d = plane2d_its(d_plane, o_plane);
                let p2_2d = plane2d_its(-d_plane, o_plane);
                //dbg!(sample_alpha, &o_plane,&d_plane,&p1_2d, &p2_2d);

                // Convert them into light coordinates
                let p1 = light.o + p1_2d.x * light.u + p1_2d.y * light.v;
                let p2 = light.o + p2_2d.x * light.u + p2_2d.y * light.v;
                let u_plane = p2 - p1;
                let u_plane_length = u_plane.magnitude();
                let u_plane = u_plane / u_plane_length;
                SinglePhotonPlane {
                    o: p1,
                    d0: u_plane,
                    d1: d,
                    length0: u_plane_length,
                    length1: t_sampled,
                    sample,
                    plane_type: PlaneType::UAlphaT,
                    // f() / (p(w_l) * p(x) * p(alpha | x))
                    // TODO: Check this formula
                    weight: std::f32::consts::PI * light.emission * (light.u_l * light.v_l)
                        / u_plane_length,
                    id_emitter,
                    sample_alpha,
                }
            }
        }
    }
}

#[derive(PartialEq)]
pub enum SinglePlaneStrategy {
    UV,
    VT,
    UT,
    Average,
    DiscreteMIS,
    UAlpha,
    ContinousMIS,
}

pub struct IntegratorSinglePlane {
    pub nb_primitive: usize,
    pub strategy: SinglePlaneStrategy,
}

impl Integrator for IntegratorSinglePlane {
    fn compute(
        &mut self,
        sampler: &mut dyn Sampler,
        accel: &dyn Acceleration,
        scene: &Scene,
    ) -> BufferCollection {
        if scene.volume.is_none() {
            panic!("Volume integrator need a volume (add -m )");
        }
        // Extract the light source
        let emitters = scene
            .meshes
            .iter()
            .filter(|m| m.is_light())
            .collect::<Vec<_>>();
        let rect_lights = {
            emitters
                .iter()
                .map(|emitter| {
                    info!("Emitter vertices: {:?}", emitter.vertices);
                    info!("Emitter indices: {:?}", emitter.indices);
                    if emitter.vertices.len() != 3 && emitter.indices.len() != 2 {
                        panic!("Only support rectangular emitters");
                    }
                    RectangularLightSource::from_shape(emitter)
                })
                .collect::<Vec<_>>()
        };

        let generate_plane = |t: PlaneType,
                              light: &[RectangularLightSource],
                              id_emitter: usize,
                              sampler: &mut dyn Sampler,
                              m: &HomogenousVolume|
         -> SinglePhotonPlane {
            let d = {
                let mut d_out = cosine_sample_hemisphere(sampler.next2d());
                while d_out.z == 0.0 {
                    // Start to generate a new plane again
                    d_out = cosine_sample_hemisphere(sampler.next2d());
                }
                let frame = Frame::new(light[id_emitter].n);
                frame.to_world(d_out)
            };

            // FIXME: Faking position as it is not important for sampling the transmittance
            let ray_med = Ray::new(light[id_emitter].o, d);
            // TODO: Check if it is the code
            // Need to check the intersection distance iff need to failed ...
            // ray_med.tfar = intersection_distance;
            let mrec = m.sample(&ray_med, sampler.next());

            // Sample planes
            let sample = sampler.next2d();
            SinglePhotonPlane::new(
                t,
                &light[id_emitter],
                d,
                sample,
                sampler.next(),
                mrec.continued_t,
                id_emitter,
                m.sigma_s,
            )
        };

        // Create the planes
        let m = scene.volume.as_ref().unwrap();
        let mut planes = vec![];
        let mut number_plane_gen = 0;
        while planes.len() < self.nb_primitive {
            let id_emitter = (sampler.next() * rect_lights.len() as f32) as usize;
            match self.strategy {
                SinglePlaneStrategy::UT => planes.push(generate_plane(
                    PlaneType::UT,
                    &rect_lights,
                    id_emitter,
                    sampler,
                    m,
                )),
                SinglePlaneStrategy::VT => planes.push(generate_plane(
                    PlaneType::VT,
                    &rect_lights,
                    id_emitter,
                    sampler,
                    m,
                )),
                SinglePlaneStrategy::UV => planes.push(generate_plane(
                    PlaneType::UV,
                    &rect_lights,
                    id_emitter,
                    sampler,
                    m,
                )),
                SinglePlaneStrategy::DiscreteMIS | SinglePlaneStrategy::Average => {
                    // Generate 3 planes
                    planes.push(generate_plane(
                        PlaneType::UV,
                        &rect_lights,
                        id_emitter,
                        sampler,
                        m,
                    ));
                    planes.push(generate_plane(
                        PlaneType::VT,
                        &rect_lights,
                        id_emitter,
                        sampler,
                        m,
                    ));
                    planes.push(generate_plane(
                        PlaneType::UT,
                        &rect_lights,
                        id_emitter,
                        sampler,
                        m,
                    ));
                }
                SinglePlaneStrategy::UAlpha | SinglePlaneStrategy::ContinousMIS => {
                    planes.push(generate_plane(
                        PlaneType::UAlphaT,
                        &rect_lights,
                        id_emitter,
                        sampler,
                        m,
                    ));
                }
            }

            number_plane_gen += 1;
        }

        // Build the BVH to speedup the computation...
        let bvh_plane = BHVAccel::create(planes);

        // Generate the image block to get VPL efficiently
        let buffernames = vec![String::from("primal")];
        let mut image_blocks = generate_img_blocks(scene, sampler, &buffernames);

        // Gathering all planes
        info!("Gathering Single planes...");
        let progress_bar = Mutex::new(ProgressBar::new(image_blocks.len() as u64));
        let pool = generate_pool(scene);
        let phase_function = PhaseFunction::Isotropic();
        pool.install(|| {
            image_blocks
                .par_iter_mut()
                .for_each(|(im_block, sampler_ray)| {
                    for ix in 0..im_block.size.x {
                        for iy in 0..im_block.size.y {
                            for _ in 0..scene.nb_samples {
                                let (ix_c, iy_c) = (ix + im_block.pos.x, iy + im_block.pos.y);
                                let pix = Point2::new(
                                    ix_c as f32 + sampler_ray.next(),
                                    iy_c as f32 + sampler_ray.next(),
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
                                for (plane_its, b_id) in bvh_plane.gather(&ray) {
                                    let plane = &bvh_plane.elements[b_id];
                                    // This code is if we do not use BVH
                                    // for plane in &planes {
                                    // let plane_its = plane.intersection(&ray);
                                    // if plane_its.is_none() {
                                    // 	continue;
                                    // }
                                    // let plane_its = plane_its.unwrap();

                                    let p_hit = ray.o + ray.d * plane_its.t_cam;
                                    let p_light = plane
                                        .light_position(&rect_lights[plane.id_emitter], &plane_its);
                                    let rect_light = &rect_lights[plane.id_emitter];
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
                                            | SinglePlaneStrategy::VT
                                            | SinglePlaneStrategy::UAlpha
                                            | SinglePlaneStrategy::ContinousMIS => 1.0,
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
                                                        0.0,
                                                        t_sampled,
                                                        plane.id_emitter,
                                                        m.sigma_s,
                                                    ),
                                                    SinglePhotonPlane::new(
                                                        PlaneType::UT,
                                                        &rect_light,
                                                        d,
                                                        plane.sample,
                                                        0.0,
                                                        t_sampled,
                                                        plane.id_emitter,
                                                        m.sigma_s,
                                                    ),
                                                    SinglePhotonPlane::new(
                                                        PlaneType::VT,
                                                        &rect_light,
                                                        d,
                                                        plane.sample,
                                                        0.0,
                                                        t_sampled,
                                                        plane.id_emitter,
                                                        m.sigma_s,
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

                                                let w =
                                                    planes[debug_id].contrib(&ray.d).avg().powi(-1)
                                                        / planes
                                                            .iter()
                                                            .map(|p| {
                                                                let c = p.contrib(&ray.d).avg();
                                                                if c != 0.0 && c.is_finite() {
                                                                    c.powi(-1)
                                                                } else {
                                                                    0.0
                                                                }
                                                            })
                                                            .sum::<f32>();
                                                if w.is_finite() {
                                                    w
                                                } else {
                                                    0.0
                                                }
                                            }
                                        };
                                        //assert!(w > 0.0 && w <= 1.0);

                                        // UV planes are not importance sampled (position/direction)
                                        // For other primitive, there such importance sampled approach.
                                        let flux = match self.strategy {
                                            SinglePlaneStrategy::ContinousMIS => {
                                                // Here we use their integration from
                                                // Normally, all the jacobian simplifies
                                                // So it is why we need to have a special estimator
                                                let w_cmis = 1.0
                                                    / ((2.0 / std::f32::consts::PI)
                                                        * (rect_light
                                                            .u
                                                            .cross(plane.d1)
                                                            .dot(ray.d)
                                                            .powi(2)
                                                            + rect_light
                                                                .v
                                                                .cross(plane.d1)
                                                                .dot(ray.d)
                                                                .powi(2))
                                                        .sqrt());
                                                w_cmis * plane.weight
                                            }
                                            _ => plane.contrib(&ray.d),
                                        };
                                        c += w
                                            * rho
                                            * transmittance
                                            * m.sigma_s
                                            * flux
                                            * (emitters.len() as f32)
                                            * (1.0 / number_plane_gen as f32);
                                    }
                                }

                                im_block.accumulate(
                                    Point2 { x: ix, y: iy },
                                    c,
                                    &"primal".to_owned(),
                                );
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
        for (im_block, _) in &image_blocks {
            image.accumulate_bitmap(im_block);
        }
        image
    }
}
