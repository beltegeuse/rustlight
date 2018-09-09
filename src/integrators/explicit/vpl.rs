use cgmath::{InnerSpace, Point2, Point3, Vector3};
use geometry::Mesh;
use integrators::*;
use paths::path::*;
use paths::vertex::*;
use samplers;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use structure::*;

pub struct IntegratorVPL {
    pub nb_vpl: usize,
    pub max_depth: Option<u32>,
    pub clamping_factor: Option<f32>,
}

struct VPLSurface<'a> {
    its: Intersection<'a>,
    radiance: Color,
}
struct VPLEmitter<'a> {
    mesh: &'a Arc<Mesh>,
    pos: Point3<f32>,
    n: Vector3<f32>,
    emitted_radiance: Color,
}

enum VPL<'a> {
    Surface(VPLSurface<'a>),
    Emitter(VPLEmitter<'a>),
}

pub struct TechniqueVPL {
    pub max_depth: Option<u32>,
    pub samplings: Vec<Box<SamplingStrategy>>,
    pub pdf_vertex: Option<PDF>,
}

impl<'a> Technique<'a> for TechniqueVPL {
    fn init(
        &mut self,
        scene: &'a Scene,
        sampler: &mut Sampler,
    ) -> Vec<(Rc<RefCell<Vertex<'a>>>, Color)> {
        let (mesh, pdf, sampled_point) =
            scene.random_sample_emitter_position(sampler.next(), sampler.next(), sampler.next2d());
        let emitter_vertex = Rc::new(RefCell::new(Vertex::Emitter(EmitterVertex {
            pos: sampled_point.p,
            n: sampled_point.n,
            mesh,
            edge_in: None,
            edge_out: None,
        })));
        self.pdf_vertex = Some(pdf); // Capture the pdf for later evaluation
        vec![(emitter_vertex, Color::one())]
    }

    fn expand(&self, _vertex: &Rc<RefCell<Vertex<'a>>>) -> bool {
        return true;
    }

    fn strategies(&self, _vertex: &Rc<RefCell<Vertex<'a>>>) -> &Vec<Box<SamplingStrategy>> {
        &self.samplings
    }
}

impl TechniqueVPL {
    fn convert_vpl<'a>(
        &self,
        scene: &'a Scene,
        vertex: &Rc<VertexPtr<'a>>,
        vpls: &RefCell<Vec<Box<VPL<'a>>>>,
        flux: Color,
    ) {
        match *vertex.borrow() {
            Vertex::Surface(ref v) => {
                vpls.borrow_mut().push(Box::new(VPL::Surface(VPLSurface {
                    its: v.its.clone(),
                    radiance: flux,
                })));

                for edge in &v.edge_out {
                    let edge = edge.borrow();
                    if let Some(ref vertex_next) = edge.vertices.1 {
                        self.convert_vpl(
                            scene,
                            vertex_next,
                            vpls,
                            flux * edge.weight * edge.rr_weight,
                        );
                    }
                }
            }
            Vertex::Emitter(ref v) => {
                let flux = v.mesh.emission / self.pdf_vertex.as_ref().unwrap().value();
                vpls.borrow_mut().push(Box::new(VPL::Emitter(VPLEmitter {
                    mesh: v.mesh,
                    pos: v.pos,
                    n: v.n,
                    emitted_radiance: flux,
                })));

                if let Some(ref edge) = v.edge_out {
                    let edge = edge.borrow();
                    if let Some(ref next_vertex) = edge.vertices.1 {
                        self.convert_vpl(scene, next_vertex, vpls, edge.weight * flux);
                    }
                }
            }
            _ => {}
        }
    }
}

impl Integrator for IntegratorVPL {
    fn compute(&mut self, scene: &Scene) -> Bitmap {
        info!("Generating the VPL...");
        let buffernames = vec!["primal".to_string()];
        let mut sampler = samplers::independent::IndependentSampler::default();
        let mut nb_path_shot = 0;
        let vpls = RefCell::new(vec![]);
        while vpls.borrow().len() < self.nb_vpl as usize {
            let samplings: Vec<Box<SamplingStrategy>> =
                vec![Box::new(DirectionalSamplingStrategy {})];
            let mut technique = TechniqueVPL {
                max_depth: self.max_depth,
                samplings,
                pdf_vertex: None,
            };

            let root = generate(scene, &mut sampler, &mut technique);
            technique.convert_vpl(scene, &root[0].0, &vpls, Color::one());
            nb_path_shot += 1;
        }
        let vpls = vpls.into_inner();

        // Generate the image block to get VPL efficiently
        let mut image_blocks: Vec<Bitmap> = Vec::new();
        for ix in StepRangeInt::new(0, scene.camera.size().x as usize, 16) {
            for iy in StepRangeInt::new(0, scene.camera.size().y as usize, 16) {
                let mut block = Bitmap::new(
                    Point2 {
                        x: ix as u32,
                        y: iy as u32,
                    },
                    Vector2 {
                        x: cmp::min(16, scene.camera.size().x - ix as u32),
                        y: cmp::min(16, scene.camera.size().y - iy as u32),
                    },
                    &buffernames,
                );
                image_blocks.push(block);
            }
        }

        // Render the image blocks VPL integration
        info!("Gathering VPL...");
        let progress_bar = Mutex::new(ProgressBar::new(image_blocks.len() as u64));
        let pool = generate_pool(scene);
        pool.install(|| {
            image_blocks.par_iter_mut().for_each(|im_block| {
                let mut sampler = independent::IndependentSampler::default();
                for ix in 0..im_block.size.x {
                    for iy in 0..im_block.size.y {
                        for _ in 0..scene.nb_samples() {
                            let c = self.compute_vpl_contrib(
                                (ix + im_block.pos.x, iy + im_block.pos.y),
                                scene,
                                &mut sampler,
                                &vpls,
                            );
                            im_block.accumulate(Point2 { x: ix, y: iy }, c, &"primal".to_string());
                        }
                    }
                }
                im_block.scale(1.0 / (scene.nb_samples() as f32));
                {
                    progress_bar.lock().unwrap().inc();
                }
            });
        });

        // Fill the image
        let mut image = Bitmap::new(Point2::new(0, 0), *scene.camera.size(), &buffernames);
        for im_block in &image_blocks {
            image.accumulate_bitmap(im_block);
        }
        image.scale(1.0 / nb_path_shot as f32);
        image
    }
}

impl IntegratorVPL {
    fn compute_vpl_contrib<'a>(
        &self,
        (ix, iy): (u32, u32),
        scene: &'a Scene,
        sampler: &mut Sampler,
        vpls: &Vec<Box<VPL<'a>>>,
    ) -> Color {
        let pix = Point2::new(ix as f32 + sampler.next(), iy as f32 + sampler.next());
        let ray = scene.camera.generate(pix);
        let mut l_i = Color::zero();

        // Check if we have a intersection with the primary ray
        let its = match scene.trace(&ray) {
            Some(x) => x,
            None => return l_i,
        };

        for vpl in vpls {
            match **vpl {
                VPL::Emitter(ref vpl) => {
                    if scene.visible(&vpl.pos, &its.p) {
                        let mut d = vpl.pos - its.p;
                        let dist = d.magnitude();
                        d /= dist;

                        let emitted_radiance = vpl.mesh.emission * vpl.n.dot(-d).max(0.0);
                        let bsdf_val = its.mesh.bsdf.eval(&its.uv, &its.wi, &its.to_local(&d));
                        l_i += emitted_radiance * bsdf_val / (dist * dist);
                    }
                }
                VPL::Surface(ref vpl) => {
                    if scene.visible(&vpl.its.p, &its.p) {
                        let mut d = vpl.its.p - its.p;
                        let dist = d.magnitude();
                        d /= dist;

                        let emitted_radiance = vpl.its.mesh.bsdf.eval(
                            &vpl.its.uv,
                            &vpl.its.wi,
                            &vpl.its.to_local(&-d),
                        );
                        let bsdf_val = its.mesh.bsdf.eval(&its.uv, &its.wi, &its.to_local(&d));
                        l_i += emitted_radiance * bsdf_val * vpl.radiance / (dist * dist);
                    }
                }
            }
        }

        l_i
    }
}
