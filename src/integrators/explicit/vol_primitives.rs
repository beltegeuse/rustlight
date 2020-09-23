use crate::accel::*;
use crate::integrators::*;
use crate::paths::path::*;
use crate::paths::strategies::*;
use crate::paths::vertex::*;
use crate::structure::AABB;
use crate::volume::*;
use cgmath::{EuclideanSpace, InnerSpace, Point2, Point3, Vector3};

pub enum VolPrimitivies {
    BRE,
    Beams,
    Planes,
    VRL,
}

pub struct IntegratorVolPrimitives {
    pub nb_primitive: usize,
    pub max_depth: Option<u32>,
    pub primitives: VolPrimitivies,
}

pub struct TechniqueVolPrimitives {
    pub max_depth: Option<u32>,
    pub samplings: Vec<Box<dyn SamplingStrategy>>,
}

// TODO: Need to refactor this
impl Technique for TechniqueVolPrimitives {
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
impl Photon {
    pub fn contribute(&self, ray: &Ray, m: &HomogenousVolume, dist: f32) -> Color {
        // Evaluate the transmittance
        let transmittance = {
            let mut ray_tr = Ray::new(ray.o, ray.d);
            ray_tr.tfar = dist;
            m.transmittance(ray_tr)
        };

        // Evaluate the phase function
        // FIXME: Check the direction of d_in
        let phase_func = self.phase_function.eval(&(-ray.d), &self.d_in);

        // Kernel (2D in case of BRE)
        let weight = 1.0 / (std::f32::consts::PI * self.radius.powi(2));

        // Sum all values
        self.radiance * transmittance * phase_func * weight
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
    from_surface: bool,
}
struct PhotonBeamIts {
    u: f32, // Kernel
    v: f32, // Beam
    w: f32, // Camera
    sin_theta: f32,
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
        if d1d2_sqr_minus1 < 1e-5 && d1d2_sqr_minus1 > -1e-5 {
            return None;
        }
        let d1o1 = r.d.dot(r.o.to_vec());
        let d1o2 = r.d.dot(self.o.to_vec());

        // Out of range on ray 1.
        let w = (d1o1 - d1o2 - d1d2 * (self.d.dot(r.o.to_vec()) - self.d.dot(self.o.to_vec())))
            / d1d2_sqr_minus1;
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
        Some(PhotonBeamIts { u, v, w, sin_theta })
    }
}
impl PhotonBeam {
    pub fn contribute(&self, ray: &Ray, m: &HomogenousVolume, beam_its: PhotonBeamIts) -> Color {
        // Evaluate the transmittance
        let transmittance = {
            let mut ray_tr = Ray::new(ray.o, ray.d);
            ray_tr.tfar = beam_its.w;
            m.transmittance(ray_tr)
        };

        // Evaluate the phase function
        // Note that we need to add sigma_s here, as we create a new vertex
        let phase_func = m.sigma_s * self.phase_function.eval(&(-ray.d), &(-self.d));

        // Jacobian * Kernel
        let weight = (1.0 / beam_its.sin_theta) * (0.5 / self.radius);
        self.radiance * transmittance * phase_func * weight
    }

    pub fn contribute_vrl(
        &self,
        ray: &Ray,
        m: &HomogenousVolume,
        accel: &dyn Acceleration,
        sampler: &mut dyn Sampler,
    ) -> Color {
        // This code is for debugging
        // It is the naive VRL sampling
        let (t_cam, t_vrl, inv_pdf) = (
            (ray.tfar - ray.tnear) * sampler.next() + ray.tnear,
            self.length * sampler.next(),
            self.length * (ray.tfar - ray.tnear),
        );

        // TODO: Implement more advanced sampling
        // let (t_cam, t_vrl, inv_pdf) = {
        //
        // }

        // Point on sensor and vrl
        // check mutual visibility
        let p_vrl = self.o + self.d * t_vrl;
        let p_cam = ray.o + ray.d * t_cam;
        if !accel.visible(&p_cam, &p_vrl) {
            return Color::zero();
        }

        // Compute direction and distance
        let d_cam_vrl = p_vrl - p_cam;
        let dist = d_cam_vrl.magnitude();
        let d_cam_vrl = d_cam_vrl / dist;

        // Compute transmittances
        let transmittance_cam = {
            let mut ray_tr = Ray::new(ray.o, ray.d);
            ray_tr.tfar = t_cam;
            m.transmittance(ray_tr)
        };
        let transmittance_vrl = {
            let mut ray_tr = Ray::new(p_vrl, -d_cam_vrl);
            ray_tr.tfar = dist;
            m.transmittance(ray_tr)
        };

        // Phase functions
        let phase_func_vrl = m.sigma_s * self.phase_function.eval(&(-self.d), &(-d_cam_vrl));
        let phase_func_cam = m.sigma_s * self.phase_function.eval(&(-ray.d), &d_cam_vrl);

        let contrib =
            self.radiance * phase_func_vrl * phase_func_cam * transmittance_cam * transmittance_vrl;
        contrib * inv_pdf / (dist * dist)
    }
}

// ------------ Plane representation
#[derive(Debug)]
struct PhotonPlane {
    o: Point3<f32>,
    d0: Vector3<f32>,
    d1: Vector3<f32>,
    length0: f32,
    length1: f32,
    phase_function: PhaseFunction,
    radiance: Color,
}
#[derive(Debug)]
struct PhotonPlaneIts {
    t_cam: f32,
    t0: f32,
    t1: f32,
    inv_det: f32,
}
impl BVHElement<PhotonPlaneIts> for PhotonPlane {
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
impl PhotonPlane {
    pub fn contribute(
        &self,
        accel: &dyn Acceleration,
        ray: &Ray,
        m: &HomogenousVolume,
        plane_its: PhotonPlaneIts,
    ) -> Color {
        let p_its = ray.o + ray.d * plane_its.t_cam;
        let p0 = self.o + self.d0 * plane_its.t0;

        // Check visibility
        if !accel.visible(&p0, &p_its) {
            return Color::zero();
        }

        // Evaluate transmittance from camera
        // Note that the other transmittance on the photon plane
        // Do not need to be evaluate as its is a short-short implementation
        let transmittance = {
            let mut ray_tr = Ray::new(ray.o, ray.d);
            ray_tr.tfar = plane_its.t_cam;
            m.transmittance(ray_tr)
        };

        // Phase functions
        // Note that only the phase function need to be evaluated at the intersection point
        // The rest of the intersections are importance sampled (so cancel out)
        // Only the sigma_s remains, so we add them at the end...
        let phase_func = self.phase_function.eval(&(-ray.d), &(-self.d1));

        // Jacobian from the paper
        // TODO: Normally it is contain inside the intersection
        let inv_jacobian = 1.0 / self.d0.dot(self.d1.cross(-ray.d)).abs();

        // Note that we do not have kernel terns as it is a zero kernel size...
        self.radiance * phase_func * m.sigma_s * m.sigma_s * transmittance * inv_jacobian
    }
}

impl TechniqueVolPrimitives {
    fn convert_planes<'scene>(
        &self,
        path: &Path<'scene>,
        scene: &'scene Scene,
        vertex_id: VertexID,
        planes: &mut Vec<PhotonPlane>,
        flux: Color,
    ) {
        match path.vertex(vertex_id) {
            Vertex::Volume { edge_out, pos, .. } => {
                for edge in edge_out {
                    let edge = path.edge(*edge);
                    if let Some(vertex_next_id) = edge.vertices.1 {
                        // Need to check two things:
                        //  1) There is one extra edge
                        //  2) The next vertex is not on the surface
                        if !path.vertex(vertex_next_id).on_surface()
                            && path.have_next_vertices(vertex_next_id)
                        {
                            // TODO: Note that we have only one edge on the next path...
                            assert_eq!(path.next_vertices(vertex_next_id).len(), 1);
                            let (next_edge_id, _next_next_vertex_id) =
                                path.next_vertices(vertex_next_id)[0];
                            let next_edge = path.edge(next_edge_id);

                            let length0 = edge.sampled_distance.as_ref().unwrap().continued_t;
                            let length1 = next_edge.sampled_distance.as_ref().unwrap().continued_t;
                            planes.push(PhotonPlane {
                                o: *pos,
                                d0: edge.d,
                                d1: next_edge.d,
                                length0,
                                length1,
                                phase_function: PhaseFunction::Isotropic(),
                                radiance: flux,
                            });
                        }
                    }
                }
            }
            Vertex::Light { .. } => {
                // Flux already have light flux
            }
            _ => {
                // Do nothing except for the medium
            }
        }

        for (edge_id, next_vertex_id) in path.next_vertices(vertex_id) {
            let edge = path.edge(edge_id);
            self.convert_planes(
                path,
                scene,
                next_vertex_id,
                planes,
                flux * edge.weight * edge.rr_weight,
            );
        }
    }

    fn convert_beams<'scene>(
        &self,
        only_from_surface: bool,
        path: &Path<'scene>,
        scene: &'scene Scene,
        vertex_id: VertexID,
        beams: &mut Vec<PhotonBeam>,
        radius: f32,
        flux: Color,
    ) {
        match path.vertex(vertex_id) {
            Vertex::Surface { edge_out, its, .. } => {
                for edge in edge_out {
                    let edge = path.edge(*edge);
                    if let Some(_vertex_next_id) = edge.vertices.1 {
                        // Always push this as it come from surfaces
                        beams.push(PhotonBeam {
                            o: its.p,
                            d: edge.d,
                            length: edge.dist.unwrap(),
                            phase_function: PhaseFunction::Isotropic(),
                            radiance: flux,
                            radius,
                            from_surface: true,
                        });
                    }
                }
            }
            Vertex::Volume { edge_out, pos, .. } => {
                for edge in edge_out {
                    let edge = path.edge(*edge);
                    if let Some(vertex_next_id) = edge.vertices.1 {
                        // Need to check two things (inverse)
                        //  1) There is one extra edge
                        //  2) The next vertex is not on the surface
                        let push_beam = if only_from_surface {
                            path.vertex(vertex_next_id).on_surface()
                                || !path.have_next_vertices(vertex_next_id)
                        } else {
                            true // Push all the vertices
                        };
                        if push_beam {
                            beams.push(PhotonBeam {
                                o: *pos,
                                d: edge.d,
                                length: edge.dist.unwrap(),
                                phase_function: PhaseFunction::Isotropic(),
                                radiance: flux,
                                radius,
                                from_surface: false,
                            });
                        }
                    }
                }
            }
            Vertex::Light { edge_out, pos, .. } => {
                if let Some(edge) = edge_out {
                    let edge = path.edge(*edge);
                    if let Some(_next_vertex_id) = edge.vertices.1 {
                        beams.push(PhotonBeam {
                            o: *pos,
                            d: edge.d,
                            length: edge.dist.unwrap(),
                            phase_function: PhaseFunction::Isotropic(),
                            radiance: flux,
                            radius,
                            from_surface: true,
                        });
                    }
                }
            }
            Vertex::Sensor { .. } => {}
        }

        for (edge_id, next_vertex_id) in path.next_vertices(vertex_id) {
            let edge = path.edge(edge_id);
            self.convert_beams(
                only_from_surface,
                path,
                scene,
                next_vertex_id,
                beams,
                radius,
                flux * edge.weight * edge.rr_weight,
            );
        }
    }

    fn convert_photons<'scene>(
        &self,
        path: &Path<'scene>,
        scene: &'scene Scene,
        vertex_id: VertexID,
        photons: &mut Vec<Photon>,
        radius: f32,
        flux: Color,
    ) {
        match path.vertex(vertex_id) {
            Vertex::Surface { .. } => {}
            Vertex::Volume {
                pos,
                d_in,
                phase_function,
                ..
            } => {
                photons.push(Photon {
                    pos: *pos,
                    d_in: *d_in,
                    phase_function: phase_function.clone(),
                    radiance: flux,
                    radius,
                });
            }
            Vertex::Light { .. } => {}
            Vertex::Sensor { .. } => {}
        }

        for (edge_id, next_vertex_id) in path.next_vertices(vertex_id) {
            let edge = path.edge(edge_id);
            self.convert_photons(
                path,
                scene,
                next_vertex_id,
                photons,
                radius,
                flux * edge.weight * edge.rr_weight,
            );
        }
    }
}

impl Integrator for IntegratorVolPrimitives {
    fn compute(
        &mut self,
        sampler: &mut dyn Sampler,
        accel: &dyn Acceleration,
        scene: &Scene,
    ) -> BufferCollection {
        if scene.volume.is_none() {
            panic!("Volume integrator need a volume (add -m )");
        }

        // FIXME: The max depth might be wrong in our integrator
        match self.primitives {
            VolPrimitivies::BRE => info!("Render with Beam radiance estimate"),
            VolPrimitivies::Beams => info!("Render with Photon beams"),
            VolPrimitivies::Planes => info!("Render with Photon planes"),
            VolPrimitivies::VRL => info!("Render with VRL"),
        }

        info!("Generating the light paths...");
        let buffernames = vec![String::from("primal")];
        let mut nb_path_shot = 0;

        // Primitives vectors
        let mut photons = vec![];
        let mut beams = vec![];
        let mut planes = vec![];

        let mut still_shoot = true;
        let samplings: Vec<Box<dyn SamplingStrategy>> = vec![Box::new(
            crate::paths::strategies::directional::DirectionalSamplingStrategy {
                transport: Transport::Radiance,
            },
        )];
        let mut technique = TechniqueVolPrimitives {
            max_depth: self.max_depth,
            samplings,
        };
        let mut path = Path::default();
        while still_shoot {
            path.clear();
            let root = path.from_light(scene, sampler);
            generate(&mut path, root.0, accel, scene, sampler, &mut technique);
            match self.primitives {
                VolPrimitivies::Beams | VolPrimitivies::VRL => {
                    technique.convert_beams(false, &path, scene, root.0, &mut beams, 0.001, root.1);
                    still_shoot = beams.len() < self.nb_primitive as usize;
                }
                VolPrimitivies::BRE => {
                    technique.convert_photons(&path, scene, root.0, &mut photons, 0.001, root.1);
                    still_shoot = photons.len() < self.nb_primitive as usize;
                }
                VolPrimitivies::Planes => {
                    // Generate beams from surfaces
                    technique.convert_beams(true, &path, scene, root.0, &mut beams, 0.001, root.1);
                    // Generate planes
                    technique.convert_planes(&path, scene, root.0, &mut planes, root.1);
                    still_shoot = planes.len() < self.nb_primitive as usize;
                }
            }
            nb_path_shot += 1;
        }

        // Special case with VRL:
        // We will split the photon beams into two groups :
        //  - One from the surface (single), which be render by photon beams
        //  - One from the volume (multiple), which be render by VRL
        let (mut beams, vrls, avg_radiance_vrl) = match self.primitives {
            VolPrimitivies::VRL => {
                let (beams, vrl): (Vec<PhotonBeam>, Vec<PhotonBeam>) =
                    beams.into_iter().partition(|b| b.from_surface);
                let avg_vrl_rad =
                    vrl.iter().map(|v| v.radiance.channel_max()).sum::<f32>() / vrl.len() as f32;
                (beams, Some(vrl), avg_vrl_rad)
            }
            _ => (beams, None, 0.0),
        };

        // Here we will cut the number of beams into subbeams
        // this will improve the performance of the BVH gathering
        // Note that for the moment it works only for short query.
        match self.primitives {
            VolPrimitivies::Beams | VolPrimitivies::Planes | VolPrimitivies::VRL => {
                const SPLIT: usize = 5; // TODO: Make as parameter if sensitive
                let avg_split =
                    beams.iter().map(|b| b.length).sum::<f32>() / (beams.len() * SPLIT) as f32;
                let nb_new_beams = beams
                    .iter()
                    .map(|b| (b.length / avg_split).ceil() as u32)
                    .sum::<u32>();
                info!(
                    "Splitting beams: {} avg size | {} new beams",
                    avg_split, nb_new_beams
                );

                let mut new_beams = vec![];
                new_beams.reserve(nb_new_beams as usize);
                for b in beams {
                    let number_split = (b.length / avg_split).ceil() as u32;
                    let new_length = b.length / number_split as f32;
                    for i in 0..number_split {
                        new_beams.push(PhotonBeam {
                            o: b.o + b.d * (i as f32) * new_length,
                            d: b.d,
                            length: new_length,
                            phase_function: b.phase_function.clone(), // FIXME: Might be wrong when deadling with spatially varying phase functions
                            radiance: b.radiance,
                            radius: b.radius,
                            from_surface: b.from_surface,
                        })
                    }
                }
                beams = new_beams;
            }
            _ => {}
        }

        // Build acceleration structure
        // Note that the radius here is fixed
        info!("Construct BVH...");
        let (bvh_photon, bvh_beams, bvh_planes) = match self.primitives {
            VolPrimitivies::Beams | VolPrimitivies::VRL => {
                (None, Some(BHVAccel::create(beams)), None)
            }
            VolPrimitivies::BRE => (Some(BHVAccel::create(photons)), None, None),
            VolPrimitivies::Planes => (
                None,
                Some(BHVAccel::create(beams)),
                Some(BHVAccel::create(planes)),
            ),
        };

        // Generate the image block to get VPL efficiently
        let mut image_blocks = generate_img_blocks(scene, sampler, &buffernames);

        // Render the image blocks VPL integration
        info!("Gathering Photons (BRE/Beams)...");
        let progress_bar = Mutex::new(ProgressBar::new(image_blocks.len() as u64));
        let norm_photon = 1.0 / nb_path_shot as f32;
        info!(" - Number of path generated: {}", nb_path_shot);
        let pool = generate_pool(scene);
        pool.install(|| {
            image_blocks.par_iter_mut().for_each(|(im_block, sampler)| {
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
                            match self.primitives {
                                VolPrimitivies::Beams => {
                                    let bvh = bvh_beams.as_ref().unwrap();
                                    for (beam_its, b_id) in bvh.gather(&ray) {
                                        c += bvh.elements[b_id].contribute(&ray, m, beam_its)
                                            * norm_photon;
                                    }
                                }
                                VolPrimitivies::VRL => {
                                    // Form surfaces only
                                    let bvh = bvh_beams.as_ref().unwrap();
                                    for (beam_its, b_id) in bvh.gather(&ray) {
                                        c += bvh.elements[b_id].contribute(&ray, m, beam_its)
                                            * norm_photon;
                                    }
                                    // Multiple-scattering
                                    for vrl in vrls.as_ref().unwrap() {
                                        // TODO: Hard-coded RR (1 VRL for 100 beams)
                                        let rr = ((vrl.radiance.channel_max() / avg_radiance_vrl)
                                            * 0.01)
                                            .min(1.0);
                                        if rr >= sampler.next() {
                                            c += (vrl.contribute_vrl(
                                                &ray,
                                                m,
                                                accel,
                                                sampler.as_mut(),
                                            ) / rr)
                                                * norm_photon;
                                        }
                                    }
                                }
                                VolPrimitivies::BRE => {
                                    let bvh = bvh_photon.as_ref().unwrap();
                                    for (dist, p_id) in bvh.gather(&ray) {
                                        c += bvh.elements[p_id].contribute(&ray, m, dist)
                                            * norm_photon;
                                    }
                                }
                                VolPrimitivies::Planes => {
                                    let bvh = bvh_beams.as_ref().unwrap();
                                    for (beam_its, b_id) in bvh.gather(&ray) {
                                        c += bvh.elements[b_id].contribute(&ray, m, beam_its)
                                            * norm_photon;
                                    }
                                    let bvh = bvh_planes.as_ref().unwrap();
                                    for (plane_its, b_id) in bvh.gather(&ray) {
                                        c += bvh.elements[b_id]
                                            .contribute(accel, &ray, m, plane_its)
                                            * norm_photon;
                                    }
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
        for (im_block, _) in &image_blocks {
            image.accumulate_bitmap(im_block);
        }
        image
    }
}
