use crate::emitter::Emitter;
use crate::integrators::*;
use crate::paths::path::*;
use crate::paths::strategies::*;
use crate::paths::vertex::*;
use crate::volume::*;
use cgmath::{EuclideanSpace, InnerSpace, Point2, Point3, Vector3};

#[derive(PartialEq, Clone)]
pub enum IntegratorVPLOption {
    Volume,
    Surface,
    All,
}

pub struct IntegratorVPL {
    pub nb_vpl: usize,
    pub max_depth: Option<u32>,
    pub rr_depth: Option<u32>,
    pub clamping_factor: Option<f32>,
    pub option_vpl: IntegratorVPLOption,
    pub option_lt: IntegratorVPLOption,
}

struct VPLSurface<'a> {
    its: Intersection<'a>,
    radiance: Color,
}
struct VPLVolume {
    pos: Point3<f32>,
    d_in: Vector3<f32>,
    phase_function: PhaseFunction,
    radiance: Color,
}

enum VPLEmitterInfo {
    Position { pos: Point3<f32>, n: Vector3<f32> },
    Infinite { d: Vector3<f32> },
}
struct VPLEmitter {
    info: VPLEmitterInfo,
    emitted_radiance: Color,
}

enum VPL<'a> {
    Surface(VPLSurface<'a>),
    Volume(VPLVolume),
    Emitter(VPLEmitter),
}

pub struct TechniqueVPL {
    pub max_depth: Option<u32>,
    pub samplings: Vec<Box<dyn SamplingStrategy>>,
}

impl Technique for TechniqueVPL {
    fn expand(&self, _vertex: &Vertex, depth: u32) -> bool {
        self.max_depth.map_or(true, |max| depth < max)
    }

    fn strategies(&self, _vertex: &Vertex) -> &Vec<Box<dyn SamplingStrategy>> {
        &self.samplings
    }
}

impl TechniqueVPL {
    fn convert_vpl<'scene>(
        &self,
        path: &Path<'scene>,
        scene: &'scene Scene,
        vertex_id: VertexID,
        options: &IntegratorVPLOption,
        vpls: &mut Vec<VPL<'scene>>,
        flux: Color,
    ) {
        match path.vertex(vertex_id) {
            Vertex::Surface { its, edge_out, .. } => {
                let bsdf_smooth = its.mesh.bsdf.bsdf_type().is_smooth();
                if *options != IntegratorVPLOption::Volume && !bsdf_smooth {
                    vpls.push(VPL::Surface(VPLSurface {
                        its: its.clone(),
                        radiance: flux,
                    }));
                }

                // Continue to bounce...
                for edge in edge_out {
                    let edge = path.edge(*edge);
                    if let Some(vertex_next_id) = edge.vertices.1 {
                        self.convert_vpl(
                            path,
                            scene,
                            vertex_next_id,
                            options,
                            vpls,
                            flux * edge.weight * edge.rr_weight,
                        );
                    }
                }
            }
            Vertex::Volume {
                pos,
                d_in,
                phase_function,
                edge_out,
                ..
            } => {
                if *options != IntegratorVPLOption::Surface {
                    vpls.push(VPL::Volume(VPLVolume {
                        pos: *pos,
                        d_in: *d_in,
                        phase_function: phase_function.clone(),
                        radiance: flux,
                    }));
                }

                // Continue to bounce...
                for edge in edge_out {
                    let edge = path.edge(*edge);
                    if let Some(vertex_next_id) = edge.vertices.1 {
                        self.convert_vpl(
                            path,
                            scene,
                            vertex_next_id,
                            options,
                            vpls,
                            flux * edge.weight * edge.rr_weight,
                        );
                    }
                }
            }
            Vertex::Light {
                edge_out, pos, n, ..
            } => {
                if *options != IntegratorVPLOption::Volume {
                    // For now it is kinda difficult to test the type
                    // We need to check the edge out...
                    let edge_out = edge_out.unwrap();
                    let edge = path.edge(edge_out);
                    match edge.pdf_direction {
                        PDF::SolidAngle(_) => {
                            vpls.push(VPL::Emitter(VPLEmitter {
                                info: VPLEmitterInfo::Position { pos: *pos, n: *n },
                                emitted_radiance: flux,
                            }));
                        }
                        PDF::Discrete(_) => vpls.push(VPL::Emitter(VPLEmitter {
                            info: VPLEmitterInfo::Infinite { d: edge.d },
                            emitted_radiance: flux,
                        })),
                        _ => todo!(),
                    }
                }

                if let Some(edge) = edge_out {
                    let edge = path.edge(*edge);
                    if let Some(next_vertex_id) = edge.vertices.1 {
                        self.convert_vpl(
                            path,
                            scene,
                            next_vertex_id,
                            options,
                            vpls,
                            edge.weight * flux * edge.rr_weight,
                        );
                    }
                }
            }
            Vertex::Sensor { .. } => {}
        }
    }
}

impl Integrator for IntegratorVPL {
    fn compute(
        &mut self,
        sampler: &mut dyn Sampler,
        accel: &dyn Acceleration,
        scene: &Scene,
    ) -> BufferCollection {
        info!("Generating the VPL...");
        let buffernames = vec![String::from("primal")];
        let mut nb_path_shot = 0;
        let mut vpls = vec![];

        // Samplings
        let samplings: Vec<Box<dyn SamplingStrategy>> = vec![Box::new(
            crate::paths::strategies::directional::DirectionalSamplingStrategy {
                transport: Transport::Radiance,
                rr_depth: self.rr_depth,
            },
        )];
        let mut technique = TechniqueVPL {
            max_depth: self.max_depth,
            samplings,
        };
        let mut path = Path::default();
        while vpls.len() < self.nb_vpl as usize {
            path.clear();
            let root = path.from_light(scene, sampler);
            generate(&mut path, root.0, accel, scene, sampler, &mut technique);
            technique.convert_vpl(&path, scene, root.0, &self.option_vpl, &mut vpls, root.1);
            nb_path_shot += 1;
        }
        let vpls = vpls;

        // Generate the image block to get VPL efficiently
        let mut image_blocks = generate_img_blocks(scene, sampler, &buffernames);

        // Render the image blocks VPL integration
        info!("Gathering VPL...");
        let progress_bar = Mutex::new(ProgressBar::new(image_blocks.len() as u64));
        let norm_vpl = 1.0 / nb_path_shot as f32;
        let pool = generate_pool(scene);
        pool.install(|| {
            image_blocks.par_iter_mut().for_each(|(im_block, sampler)| {
                for ix in 0..im_block.size.x {
                    for iy in 0..im_block.size.y {
                        for _ in 0..scene.nb_samples {
                            let c = self.compute_vpl_contrib(
                                (ix + im_block.pos.x, iy + im_block.pos.y),
                                accel,
                                scene,
                                sampler.as_mut(),
                                &vpls,
                                norm_vpl,
                            );
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

impl IntegratorVPL {
    fn transmittance(
        &self,
        medium: Option<&HomogenousVolume>,
        p1: Point3<f32>,
        p2: Point3<f32>,
    ) -> Color {
        if let Some(m) = medium {
            let mut d = p2 - p1;
            let dist = d.magnitude();
            d /= dist;
            let mut r = Ray::new(p1, d);
            r.tfar = dist;
            m.transmittance(r)
        } else {
            Color::one()
        }
    }

    fn gathering_surface<'a>(
        &self,
        medium: Option<&HomogenousVolume>,
        accel: &dyn Acceleration,
        vpls: &[VPL<'a>],
        norm_vpl: f32,
        its: &Intersection,
    ) -> Color {
        let mut l_i = Color::zero();

        // Self emission
        if its.cos_theta() > 0.0 {
            l_i += its.mesh.emit(&its.uv);
        }

        for vpl in vpls {
            match *vpl {
                VPL::Emitter(ref vpl) => {
                    match vpl.info {
                        VPLEmitterInfo::Position { pos, n } => {
                            if accel.visible(&pos, &its.p) {
                                let mut d = pos - its.p;
                                let dist = d.magnitude();
                                d /= dist;

                                // TODO: Check why this difference...
                                let emitted_radiance = vpl.emitted_radiance
                                    * n.dot(-d).max(0.0)
                                    * std::f32::consts::FRAC_1_PI;
                                if !its.mesh.bsdf.bsdf_type().is_smooth() {
                                    let bsdf_val = its.mesh.bsdf.eval(
                                        &its.uv,
                                        &its.wi,
                                        &its.to_local(&d),
                                        Domain::SolidAngle,
                                        Transport::Importance,
                                    );
                                    let trans = self.transmittance(medium, its.p, pos);
                                    l_i += trans * norm_vpl * emitted_radiance * bsdf_val
                                        / (dist * dist);
                                }
                            }
                        }
                        VPLEmitterInfo::Infinite { d } => {
                            let ray = Ray::spawn_ray(&its, -d);
                            if accel.trace(&ray).is_none() {
                                let emitted_radiance = vpl.emitted_radiance;
                                if !its.mesh.bsdf.bsdf_type().is_smooth() {
                                    let bsdf_val = its.mesh.bsdf.eval(
                                        &its.uv,
                                        &its.wi,
                                        &its.to_local(&-d),
                                        Domain::SolidAngle,
                                        Transport::Importance,
                                    );
                                    // TODO: Medium is not supported yet...
                                    assert!(medium.is_none());
                                    //let trans = self.transmittance(medium, its.p, pos);
                                    l_i += norm_vpl * emitted_radiance * bsdf_val;
                                    // trans *
                                }
                            }
                        }
                    }
                }
                VPL::Volume(ref vpl) => {
                    let mut d = vpl.pos - its.p;
                    let dist = d.magnitude();
                    d /= dist;

                    if !its.mesh.bsdf.bsdf_type().is_smooth() {
                        let emitted_radiance = vpl.phase_function.eval(&vpl.d_in, &d);
                        let bsdf_val = its.mesh.bsdf.eval(
                            &its.uv,
                            &its.wi,
                            &its.to_local(&d),
                            Domain::SolidAngle,
                            Transport::Importance,
                        );
                        let trans = self.transmittance(medium, its.p, vpl.pos);
                        l_i += trans * norm_vpl * emitted_radiance * bsdf_val * vpl.radiance
                            / (dist * dist);
                    }
                }
                VPL::Surface(ref vpl) => {
                    if accel.visible(&vpl.its.p, &its.p) {
                        let mut d = vpl.its.p - its.p;
                        let dist = d.magnitude();
                        d /= dist;

                        if !its.mesh.bsdf.bsdf_type().is_smooth() {
                            let emitted_radiance = vpl.its.mesh.bsdf.eval(
                                &vpl.its.uv,
                                &vpl.its.wi,
                                &vpl.its.to_local(&-d),
                                Domain::SolidAngle,
                                Transport::Radiance, // TODO: Check this
                            );
                            let bsdf_val = its.mesh.bsdf.eval(
                                &its.uv,
                                &its.wi,
                                &its.to_local(&d),
                                Domain::SolidAngle,
                                Transport::Importance,
                            );
                            let trans = self.transmittance(medium, its.p, vpl.its.p);
                            l_i += trans * norm_vpl * emitted_radiance * bsdf_val * vpl.radiance
                                / (dist * dist);
                        }
                    }
                }
            }
        }
        l_i
    }

    fn gathering_volume<'a>(
        &self,
        medium: Option<&HomogenousVolume>,
        accel: &dyn Acceleration,
        vpls: &[VPL<'a>],
        norm_vpl: f32,
        d_cam: Vector3<f32>,
        its_pos: Point3<f32>,
        phase: &PhaseFunction,
    ) -> Color {
        let mut l_i = Color::zero();
        for vpl in vpls {
            match *vpl {
                VPL::Emitter(ref vpl) => {
                    match vpl.info {
                        VPLEmitterInfo::Position { pos, n } => {
                            if accel.visible(&pos, &its_pos) {
                                let mut d = pos - its_pos;
                                let dist = d.magnitude();
                                d /= dist;

                                let emitted_radiance = vpl.emitted_radiance
                                    * n.dot(-d).max(0.0)
                                    * std::f32::consts::FRAC_1_PI;
                                let phase_val = phase.eval(&d_cam, &d);
                                let trans = self.transmittance(medium, pos, its_pos);
                                l_i +=
                                    trans * norm_vpl * emitted_radiance * phase_val / (dist * dist);
                            }
                        }
                        VPLEmitterInfo::Infinite { d } => {
                            let ray = Ray::new(its_pos, -d);
                            if accel.trace(&ray).is_none() {
                                let phase_val = phase.eval(&d_cam, &d);
                                assert!(medium.is_none());
                                // let trans = self.transmittance(medium, pos, vpl.pos);
                                let emitted_radiance = vpl.emitted_radiance;
                                l_i += norm_vpl * emitted_radiance * phase_val; // trans *
                            }
                        }
                    }
                }
                VPL::Volume(ref vpl) => {
                    let mut d = vpl.pos - its_pos;
                    let dist = d.magnitude();
                    d /= dist;

                    let emitted_radiance = vpl.phase_function.eval(&vpl.d_in, &d);
                    let phase_val = phase.eval(&d_cam, &d);
                    let trans = self.transmittance(medium, its_pos, vpl.pos);
                    l_i += trans * norm_vpl * emitted_radiance * phase_val * vpl.radiance
                        / (dist * dist);
                }
                VPL::Surface(ref vpl) => {
                    if accel.visible(&vpl.its.p, &its_pos) {
                        let mut d = vpl.its.p - its_pos;
                        let dist = d.magnitude();
                        d /= dist;

                        let emitted_radiance = vpl.its.mesh.bsdf.eval(
                            &vpl.its.uv,
                            &vpl.its.wi,
                            &vpl.its.to_local(&-d),
                            Domain::SolidAngle,
                            Transport::Radiance,
                        );
                        let phase_val = phase.eval(&d_cam, &d);
                        let trans = self.transmittance(medium, its_pos, vpl.its.p);
                        l_i += trans * norm_vpl * emitted_radiance * phase_val * vpl.radiance
                            / (dist * dist);
                    }
                }
            }
        }
        l_i
    }

    fn compute_vpl_contrib<'a>(
        &self,
        (ix, iy): (u32, u32),
        accel: &dyn Acceleration,
        scene: &'a Scene,
        sampler: &mut dyn Sampler,
        vpls: &[VPL<'a>],
        norm_vpl: f32,
    ) -> Color {
        let pix = Point2::new(ix as f32 + sampler.next(), iy as f32 + sampler.next());
        let ray = scene.camera.generate(pix);
        let mut l_i = Color::zero();

        // Check if we have a intersection with the primary ray
        let its = match accel.trace(&ray) {
            Some(x) => x,
            None => {
                if let Some(m) = &scene.volume {
                    // Sample the participating media
                    let mrec = m.sample(&ray, sampler.next());
                    assert!(!mrec.exited);
                    let pos = Point3::from_vec(ray.o.to_vec() + ray.d * mrec.t);
                    l_i *= self.gathering_volume(
                        scene.volume.as_ref(),
                        accel,
                        vpls,
                        norm_vpl,
                        -ray.d,
                        pos,
                        &m.phase,
                    ) * mrec.w;
                    return l_i;
                } else {
                    return match &scene.emitter_environment {
                        None => l_i,
                        Some(envmap) => l_i + envmap.eval(ray.d, None),
                    };
                }
            }
        };

        if let Some(m) = &scene.volume {
            let mut ray_med = ray.clone();
            ray_med.tfar = its.dist;
            let mrec = m.sample(&ray_med, sampler.next());
            if !mrec.exited {
                let pos = Point3::from_vec(ray.o.to_vec() + ray.d * mrec.t);
                l_i += self.gathering_volume(
                    scene.volume.as_ref(),
                    accel,
                    vpls,
                    norm_vpl,
                    -ray.d,
                    pos,
                    &m.phase,
                ) * mrec.w;
                l_i
            } else {
                if self.option_lt != IntegratorVPLOption::Volume {
                    l_i +=
                        self.gathering_surface(scene.volume.as_ref(), accel, vpls, norm_vpl, &its)
                            * mrec.w;
                }
                l_i
            }
        } else {
            if self.option_lt != IntegratorVPLOption::Surface {
                l_i += self.gathering_surface(scene.volume.as_ref(), accel, vpls, norm_vpl, &its);
            }
            l_i
        }
    }
}
