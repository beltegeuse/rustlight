use cgmath::{InnerSpace, Point2};
use integrators::gradient::shiftmapping::*;
use samplers::Sampler;
use std::cell::RefCell;
use std::rc::Rc;
use structure::*;

struct DiffuseReconnection {
    pub base_contrib: Color,
}
impl Default for DiffuseReconnection {
    fn default() -> Self {
        DiffuseReconnection {
            base_contrib: Color::zero(),
        }
    }
}
impl ShiftMapping for DiffuseReconnection {
    fn base<'a>(
        &mut self,
        technique: &mut TechniqueGradientPathTracing,
        pos: Point2<u32>,
        scene: &'a Scene,
        sampler: &mut Sampler,
    ) -> (Color, Rc<VertexPtr<'a>>) {
        technique.img_pos = pos;
        let root = generate(scene, sampler, technique);
        let root = root[0].0.clone();
        self.base_contrib = technique.evaluate(scene, &root);
        (self.base_contrib, root)
    }
    fn shift<'a>(
        &mut self,
        technique: &mut TechniqueGradientPathTracing,
        pos: Point2<u32>,
        scene: &Scene,
        sampler: &mut Sampler,
        base: &Rc<VertexPtr<'a>>,
    ) -> ShiftValue {
        // Generate
        technique.img_pos = pos;
        let offset = base.borrow().pixel_pos();
        let root_shift = Rc::new(RefCell::new(Vertex::Sensor(SensorVertex {
            uv: Point2::new(
                pos.x as f32 + offset.x.fract(),
                pos.y as f32 + offset.y.fract(),
            ),
            pos: scene.camera.position(),
            edge_in: None,
            edge_out: None,
        })));

        let directional = &technique.strategies(&root_shift)[0];
        if let Some((next_vertex, _next_throughput)) =
            directional.sample(root_shift.clone(), scene, Color::one(), sampler, 0)
        {
            // Checking if we have a primary point from the base path
            // if not, it is a shift failed.
            let mut primary_base = base.borrow().next_vertex();
            if primary_base.len() == 0 {
                return ShiftValue::default().base(self.base_contrib);
            }
            let primary_base = primary_base.pop().unwrap();
            self.not_reconnected(scene, &primary_base, &next_vertex);
            ShiftValue::default().base(self.base_contrib)
        } else {
            // No primary intersection from the shift path
            // this is a shift failed
            ShiftValue::default().base(self.base_contrib)
        }
    }
    fn clear(&mut self) {}
}

impl DiffuseReconnection {
    // Generate the shift path
    fn not_reconnected<'a, 'b>(
        &self,
        scene: &Scene,
        base: &Rc<VertexPtr<'b>>,
        shift: &Rc<VertexPtr<'a>>,
    ) {
        // Check the reconnection to the light source.
        for second_base in base.borrow().next_vertex() {
            match *second_base.borrow() {
                Vertex::Surface(ref main) => {
                    if main.edge_out.len() == 0 {
                        // The main path is dead, stop to do the shift mapping
                        continue;
                    }
                    let main_bsdf_pdf = 0.0; // FIXME:!!!!!!!!!!!!

                    let shift = match *shift.borrow() {
                        Vertex::Surface(ref v) => v.clone(),
                        _ => unimplemented!(),
                    };
                    if !scene.visible(&shift.its.p, &main.its.p) {
                        continue; // The shift is dead as diffuse reconnection did not work
                    }

                    // Compute the direction for diffuse reconnection
                    let shift_d_out_global = (main.its.p - shift.its.p).normalize();
                    let shift_d_out_local = shift.its.frame.to_local(shift_d_out_global);

                    // Compute jacobian of diffuse reconnection
                    let jacobian = (main.its.n_s.dot(-shift_d_out_global) * main.its.dist.powi(2))
                        .abs()
                        / (main.its.wi.z * (shift.its.p - main.its.p).magnitude2()).abs();
                    assert!(jacobian.is_finite());
                    assert!(jacobian >= 0.0);

                    // Evaluate BSDF of diffuse reconnection
                    let mut shift_bsdf_value =
                        shift
                            .its
                            .mesh
                            .bsdf
                            .eval(&shift.its.uv, &shift.its.wi, &shift_d_out_local);
                    let mut shift_bsdf_pdf = shift
                        .its
                        .mesh
                        .bsdf
                        .pdf(&shift.its.uv, &shift.its.wi, &shift_d_out_local)
                        .value();
                    
                    shift_bsdf_value *= (jacobian / main_bsdf_pdf);
                    shift_bsdf_pdf *= jacobian;

                    let ray = Ray::new(shift.its.p, shift_d_out_global);
                    // FIXME: Lifetimes
                    // let (edge, new_vertex) = Edge::from_ray(
                    //     &ray,
                    //     &base,
                    //     PDF::SolidAngle(shift_bsdf_pdf),
                    //     shift_bsdf_value,
                    //     1.0, // RR for gradient-domain
                    //     scene,
                    //     0,
                    // );
                }
                Vertex::Emitter(ref v) => {
                    // TODO: Do the explicit connection to the light
                    // TODO: The edge need to created but the contribution from the edge need to be 0
                }
                _ => panic!("Unexpected vertex"),
            }
        }
    }

    // Generate the shift path
    fn just_reconnected<'a, 'b>(&self, base: &Rc<VertexPtr<'b>>, shift: &Rc<VertexPtr<'a>>) {}

    // Generate the shift path
    fn connected<'a, 'b>(&self, base: &Rc<VertexPtr<'b>>, shift: &Rc<VertexPtr<'a>>) {
    }
}
