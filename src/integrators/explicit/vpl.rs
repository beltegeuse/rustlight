use geometry::Mesh;
use cgmath::{Point2,Vector3,InnerSpace,Point3};
use integrators::*;
use paths::path::*;
use paths::vertex::*;
use samplers;
use std::cell::RefCell;
use std::rc::Rc;
use structure::*;
use std::sync::Arc;

pub struct IntegratorVPL<'a> {
    pub nb_vpl: usize,
    pub max_depth: Option<u32>,
    pub clamping_factor: Option<f32>,
    pub vpls: Vec<Box<VPL<'a>>>,
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
        vpls: &mut Vec<Box<VPL<'a>>>,
        flux: Color,
    ) {
        match *vertex.borrow() {
            Vertex::Surface(ref v) => {
                vpls.push(Box::new(VPL::Surface(VPLSurface {
                    its: v.its,
                    radiance: flux,
                })));

                for edge in &v.edge_out {
                    let edge = edge.borrow();
                    if let Some(ref vertex_next) = edge.vertices.1 {
                        self.convert_vpl(
                            scene,
                            vertex_next,
                            &mut vpls,
                            flux * edge.weight * edge.rr_weight,
                        );
                    }
                }
            }
            Vertex::Emitter(ref v) => {
                let flux = v.mesh.emission / self.pdf_vertex.as_ref().unwrap().value();
                vpls.push(Box::new(VPL::Emitter(VPLEmitter {
                    mesh: v.mesh,
                    pos: v.pos,
                    n: v.n,
                    emitted_radiance: flux,
                })));

                if let Some(ref edge) = v.edge_out {
                    let edge = edge.borrow();
                    if let Some(ref next_vertex) = edge.vertices.1 {
                        self.convert_vpl(scene, next_vertex, &mut vpls, edge.weight * flux);
                    }
                }
            }
            _ => {}
        }
    }
}

impl<'a, 'b: 'a> Integrator for IntegratorVPL<'b> {
    fn preprocess(&mut self, scene: &Scene) {
        info!("Generating the VPL...");
        let sampler = samplers::independent::IndependentSampler::default();
        while self.vpls.len() < self.nb_vpl as usize {
            let samplings: Vec<Box<SamplingStrategy>> =
                vec![Box::new(DirectionalSamplingStrategy {})];
            let mut technique = TechniqueVPL {
                max_depth: self.max_depth,
                samplings,
                pdf_vertex: None,
            };
            let root = generate::<'b>(scene, &mut sampler, &mut technique);
            technique.convert_vpl(scene, &root[0].0, &mut self.vpls, Color::one());
        }
    }

    fn compute(&self, scene: &Scene) -> Bitmap {
        compute_mc(self, scene)
    }
}

impl<'a> IntegratorMC for IntegratorVPL<'a> {
    fn compute_pixel(&self, (ix,iy): (u32, u32), scene: &Scene, sampler: &mut Sampler) -> Color {
        let pix = Point2::new(ix as f32 + sampler.next(), iy as f32 + sampler.next());
        let mut ray = scene.camera.generate(pix);
        let mut l_i = Color::zero();
        let mut throughput = Color::one();

        // Check if we have a intersection with the primary ray
        let mut its = match scene.trace(&ray) {
            Some(x) => x,
            None => return l_i,
        };

        for vpl in &self.vpls {
            match **vpl {
                VPL::Emitter(ref vpl) => {
                    
                },
                VPL::Surface(ref vpl) => {
                }
            }
        }

        return l_i;
    }
}
