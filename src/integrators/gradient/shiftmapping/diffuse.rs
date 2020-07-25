use crate::integrators::gradient::shiftmapping::*;
use crate::paths::strategy::*;
use crate::samplers::Sampler;
use crate::structure::*;
use cgmath::Point2;

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
    fn base<'scene, 'emitter>(
        &mut self,
        path: &mut Path<'scene, 'emitter>,
        technique: &mut TechniqueGradientPathTracing,
        pos: Point2<u32>,
        accel: &'scene dyn Acceleration,
        scene: &'scene Scene,
        emitters: &'emitter EmitterSampler,
        sampler: &mut dyn Sampler,
    ) -> (Color, VertexID) {
        technique.img_pos = pos;
        let root = generate(path, accel, scene, emitters, sampler, technique);
        self.base_contrib = technique.evaluate(path, scene, emitters, root[0].0);
        (self.base_contrib, root[0].0)
    }
    fn shift<'scene, 'emitter>(
        &mut self,
        _path: &mut Path<'scene, 'emitter>,
        _technique: &mut TechniqueGradientPathTracing,
        _pos: Point2<u32>,
        _accel: &'scene dyn Acceleration,
        _scene: &'scene Scene,
        _emitters: &'emitter EmitterSampler,
        _sampler: &mut dyn Sampler,
        _base_id: VertexID,
    ) -> ShiftValue {
        unimplemented!();
        // Generate
        // technique.img_pos = pos;
        // let base = path.vertex(base_id);
        // let offset = path.vertex(base_id).pixel_pos();
        // let root_shift = Vertex::Sensor(SensorVertex {
        //     uv: Point2::new(
        //         pos.x as f32 + offset.x.fract(),
        //         pos.y as f32 + offset.y.fract(),
        //     ),
        //     pos: scene.camera.position(),
        //     edge_in: None,
        //     edge_out: None,
        // });

        // let directional = &technique.strategies(&root_shift)[0];
        // let root_shift_id = path.register_vertex(root_shift);
        // if let Some((next_vertex, _next_throughput)) =
        //     directional.sample(root_shift.clone(), scene, Color::one(), sampler, 0)
        // {
        //     // Checking if we have a primary point from the base path
        //     // if not, it is a shift failed.
        //     let mut primary_base = base.borrow().next_vertex();
        //     if primary_base.len() == 0 {
        //         return ShiftValue::default().base(self.base_contrib);
        //     }
        //     let primary_base = primary_base.pop().unwrap();
        //     self.not_reconnected(scene, &primary_base, &next_vertex);
        //     ShiftValue::default().base(self.base_contrib)
        // } else {
        //     // No primary intersection from the shift path
        //     // this is a shift failed
        //     ShiftValue::default().base(self.base_contrib)
        // }
    }
    fn clear(&mut self) {}
}

impl DiffuseReconnection {
    // Generate the shift path
    fn not_reconnected<'a, 'b>(
        &self,
        _path: &Path,
        _scene: &Scene,
        _base_id: VertexID,
        _shift_id: VertexID,
    ) {
        unimplemented!();
        // Check the reconnection to the light source.
        // for second_base in base.borrow().next_vertex() {
        //     match *second_base.borrow() {
        //         Vertex::Surface(ref main) => {
        //             if main.edge_out.len() == 0 {
        //                 // The main path is dead, stop to do the shift mapping
        //                 continue;
        //             }
        //             let main_bsdf_pdf = 0.0; // FIXME:!!!!!!!!!!!!

        //             let shift = match *shift.borrow() {
        //                 Vertex::Surface(ref v) => v.clone(),
        //                 _ => unimplemented!(),
        //             };
        //             if !scene.visible(&shift.its.p, &main.its.p) {
        //                 continue; // The shift is dead as diffuse reconnection did not work
        //             }

        //             // Compute the direction for diffuse reconnection
        //             let shift_d_out_global = (main.its.p - shift.its.p).normalize();
        //             let shift_d_out_local = shift.its.frame.to_local(shift_d_out_global);

        //             // Compute jacobian of diffuse reconnection
        //             let jacobian = (main.its.n_s.dot(-shift_d_out_global) * main.its.dist.powi(2))
        //                 .abs()
        //                 / (main.its.wi.z * (shift.its.p - main.its.p).magnitude2()).abs();
        //             assert!(jacobian.is_finite());
        //             assert!(jacobian >= 0.0);

        //             // Evaluate BSDF of diffuse reconnection
        //             assert!(!shift.its.mesh.bsdf.is_smooth());
        //             let mut shift_bsdf_value = shift.its.mesh.bsdf.eval(
        //                 &shift.its.uv,
        //                 &shift.its.wi,
        //                 &shift_d_out_local,
        //                 Domain::SolidAngle,
        //             );
        //             let mut shift_bsdf_pdf = shift
        //                 .its
        //                 .mesh
        //                 .bsdf
        //                 .pdf(
        //                     &shift.its.uv,
        //                     &shift.its.wi,
        //                     &shift_d_out_local,
        //                     Domain::SolidAngle,
        //                 )
        //                 .value();

        //             shift_bsdf_value *= jacobian / main_bsdf_pdf;
        //             shift_bsdf_pdf *= jacobian;

        //             let _ray = Ray::new(shift.its.p, shift_d_out_global);
        //             // FIXME: Lifetimes
        //             // let (edge, new_vertex) = Edge::from_ray(
        //             //     &ray,
        //             //     &base,
        //             //     PDF::SolidAngle(shift_bsdf_pdf),
        //             //     shift_bsdf_value,
        //             //     1.0, // RR for gradient-domain
        //             //     scene,
        //             //     0,
        //             // );
        //         }
        //         Vertex::Light(ref _v) => {
        //             // TODO: Do the explicit connection to the light
        //             // TODO: The edge need to created but the contribution from the edge need to be 0
        //         }
        //         _ => panic!("Unexpected vertex"),
        //     }
        // }
    }

    // Generate the shift path
    fn just_reconnected(&self, _path: &Path, _base: VertexID, _shift: VertexID) {}

    // Generate the shift path
    fn connected(&self, _path: &Path, _base: VertexID, _shift: VertexID) {}
}
