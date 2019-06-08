use crate::integrators::*;
use crate::paths::path::*;
use crate::paths::vertex::*;
use crate::samplers;
use crate::structure::*;
use cgmath::{InnerSpace, Point2, Point3, Vector3};

pub struct IntegratorVPL {
    pub nb_vpl: usize,
    pub max_depth: Option<u32>,
    pub clamping_factor: Option<f32>,
}

struct VPLSurface<'a> {
    its: Intersection<'a>,
    radiance: Color,
}
struct VPLEmitter {
    pos: Point3<f32>,
    n: Vector3<f32>,
    emitted_radiance: Color,
}

enum VPL<'a> {
    Surface(VPLSurface<'a>),
    Emitter(VPLEmitter),
}

pub struct TechniqueVPL {
    pub max_depth: Option<u32>,
    pub samplings: Vec<Box<SamplingStrategy>>,
    pub pdf_vertex: Option<PDF>,
}

impl Technique for TechniqueVPL {
    fn init<'scene, 'emitter>(
        &mut self,
        path: &mut Path<'scene, 'emitter>,
        _scene: &'scene Scene,
        sampler: &mut Sampler,
        emitters: &'emitter EmitterSampler,
    ) -> Vec<(VertexID, Color)> {
        let (emitter, sampled_point) = emitters.random_sample_emitter_position(
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
        self.pdf_vertex = Some(sampled_point.pdf); // Capture the pdf for later evaluation
        vec![(path.register_vertex(emitter_vertex), Color::one())]
    }

    fn expand(&self, _vertex: &Vertex, depth: u32) -> bool {
        self.max_depth.map_or(true, |max| depth < max)
    }

    fn strategies(&self, _vertex: &Vertex) -> &Vec<Box<SamplingStrategy>> {
        &self.samplings
    }
}

impl TechniqueVPL {
    fn convert_vpl<'scene>(
        &self,
        path: &Path<'scene, '_>,
        scene: &'scene Scene,
        vertex_id: VertexID,
        vpls: &mut Vec<VPL<'scene>>,
        flux: Color,
    ) {
        match path.vertex(vertex_id) {
            Vertex::Surface(ref v) => {
                vpls.push(VPL::Surface(VPLSurface {
                    its: v.its.clone(),
                    radiance: flux,
                }));

                for edge in &v.edge_out {
                    let edge = path.edge(*edge);
                    if let Some(vertex_next_id) = edge.vertices.1 {
                        self.convert_vpl(
                            path,
                            scene,
                            vertex_next_id,
                            vpls,
                            flux * edge.weight * edge.rr_weight,
                        );
                    }
                }
            }
            Vertex::Light(ref v) => {
                let flux = v.emitter.flux() / self.pdf_vertex.as_ref().unwrap().value();
                vpls.push(VPL::Emitter(VPLEmitter {
                    pos: v.pos,
                    n: v.n,
                    emitted_radiance: flux,
                }));

                if let Some(edge) = v.edge_out {
                    let edge = path.edge(edge);
                    if let Some(next_vertex_id) = edge.vertices.1 {
                        self.convert_vpl(path, scene, next_vertex_id, vpls, edge.weight * flux);
                    }
                }
            }
            _ => {}
        }
    }
}

impl Integrator for IntegratorVPL {
    fn compute(&mut self, scene: &Scene) -> BufferCollection {
        info!("Generating the VPL...");
        let buffernames = vec![String::from("primal")];
        let mut sampler = samplers::independent::IndependentSampler::default();
        let mut nb_path_shot = 0;
        let mut vpls = vec![];
        let emitters = scene.emitters_sampler();
        while vpls.len() < self.nb_vpl as usize {
            let samplings: Vec<Box<SamplingStrategy>> =
                vec![Box::new(DirectionalSamplingStrategy {})];
            let mut technique = TechniqueVPL {
                max_depth: self.max_depth,
                samplings,
                pdf_vertex: None,
            };
            let mut path = Path::default();
            let root = generate(&mut path, scene, &emitters, &mut sampler, &mut technique);
            technique.convert_vpl(&path, scene, root[0].0, &mut vpls, Color::one());
            nb_path_shot += 1;
        }
        let vpls = vpls;

        // Generate the image block to get VPL efficiently
        let mut image_blocks = generate_img_blocks(scene, &buffernames);

        // Render the image blocks VPL integration
        info!("Gathering VPL...");
        let progress_bar = Mutex::new(ProgressBar::new(image_blocks.len() as u64));
        let norm_vpl = 1.0 / nb_path_shot as f32;
        let pool = generate_pool(scene);
        pool.install(|| {
            image_blocks.par_iter_mut().for_each(|im_block| {
                let mut sampler = independent::IndependentSampler::default();
                for ix in 0..im_block.size.x {
                    for iy in 0..im_block.size.y {
                        for _ in 0..scene.nb_samples {
                            let c = self.compute_vpl_contrib(
                                (ix + im_block.pos.x, iy + im_block.pos.y),
                                scene,
                                &mut sampler,
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
        for im_block in &image_blocks {
            image.accumulate_bitmap(im_block);
        }
        image
    }
}

impl IntegratorVPL {
    fn compute_vpl_contrib<'a>(
        &self,
        (ix, iy): (u32, u32),
        scene: &'a Scene,
        sampler: &mut Sampler,
        vpls: &[VPL<'a>],
        norm_vpl: f32,
    ) -> Color {
        let pix = Point2::new(ix as f32 + sampler.next(), iy as f32 + sampler.next());
        let ray = scene.camera.generate(pix);
        let mut l_i = Color::zero();

        // Check if we have a intersection with the primary ray
        let its = match scene.trace(&ray) {
            Some(x) => x,
            None => return l_i,
        };

        // Check if we are on a light
        if its.cos_theta() > 0.0 {
            l_i += &(its.mesh.emission);
        }

        for vpl in vpls {
            match *vpl {
                VPL::Emitter(ref vpl) => {
                    if scene.visible(&vpl.pos, &its.p) {
                        let mut d = vpl.pos - its.p;
                        let dist = d.magnitude();
                        d /= dist;

                        let emitted_radiance = vpl.emitted_radiance * vpl.n.dot(-d).max(0.0);
                        if !its.mesh.bsdf.is_smooth() {
                            let bsdf_val = its.mesh.bsdf.eval(
                                &its.uv,
                                &its.wi,
                                &its.to_local(&d),
                                Domain::SolidAngle,
                            );
                            l_i += norm_vpl * emitted_radiance * bsdf_val / (dist * dist);
                        }
                    }
                }
                VPL::Surface(ref vpl) => {
                    if scene.visible(&vpl.its.p, &its.p) {
                        let mut d = vpl.its.p - its.p;
                        let dist = d.magnitude();
                        d /= dist;

                        if !its.mesh.bsdf.is_smooth() {
                            let emitted_radiance = vpl.its.mesh.bsdf.eval(
                                &vpl.its.uv,
                                &vpl.its.wi,
                                &vpl.its.to_local(&-d),
                                Domain::SolidAngle,
                            );
                            let bsdf_val = its.mesh.bsdf.eval(
                                &its.uv,
                                &its.wi,
                                &its.to_local(&d),
                                Domain::SolidAngle,
                            );
                            l_i += norm_vpl * emitted_radiance * bsdf_val * vpl.radiance
                                / (dist * dist);
                        }
                    }
                }
            }
        }

        l_i
    }
}
