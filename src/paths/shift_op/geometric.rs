// FIXME: This op is not ready yet
// enum ShiftGeometricState {
//     NotConnected,
//     RecentlyConnected,
//     Connected,
// }
// pub struct ShiftGeomOp {}
// impl<'a> ShiftOp<'a> for ShiftGeomOp {
//     fn generate_base<S: Sampler>(
//         &mut self,
//         (ix, iy): (u32, u32),
//         scene: &'a Scene,
//         sampler: &mut S,
//         max_depth: Option<u32>,
//     ) -> Option<Path<'a>> {
//         Path::from_sensor((ix, iy), scene, sampler, max_depth)
//     }

//     fn shift<S: Sampler>(
//         &mut self,
//         base_path: Path<'a>,
//         shift_pix: Point2<f32>,
//         scene: &'a Scene,
//         sampler: &mut S,
//         max_depth: Option<u32>,
//     ) -> Option<Path<'a>> {
//         // FIXME: Need to implement G-PT shift mapping
//         // FIXME: The idea of this code is to shift the path geometry
//         // FIXME: without evaluating the direct lighting (compared to G-PT)
//         let mut v0 = Vertex::new_sensor_vertex(shift_pix, scene.camera.param.pos);
//         let (e0, v1) = match v0.generate_next::<::sampler::IndependentSampler>(scene, None) {
//             (Some(e), Some(v)) => (e, v),
//             _ => return None, // FIXME: This is not correct for now
//         };

//         let mut vertices = vec![v0, v1];
//         let mut edges = vec![e0];
//         let mut state = ShiftGeometricState::NotConnected;
//         let mut pdf = 1.0;

//         for i in 1..base_path.vertices.len() {
//             match state {
//                 ShiftGeometricState::NotConnected => {
//                     let main_currrent = match &base_path.vertices[i] {
//                         &Vertex::Surface(ref x) => x,
//                         _ => panic!("Wrong main_current vertex type"),
//                     };
//                     let main_bsdf_pdf = match main_currrent.sampled_bsdf.as_ref().unwrap().pdf {
//                         PDF::SolidAngle(x) => x,
//                         _ => panic!("main_bsdf_pdf is not in solid angle"),
//                     };

//                     match base_path.vertices.get(i + 1) {
//                         //FIXME: Are we sure about that? Because the path might be not
//                         //FIXME: Because a edge of the path can be missing
//                         None => return Some(Path { vertices, edges }),
//                         Some(&Vertex::Surface(ref main_next)) => {
//                             let new_vertex = {
//                                 let main_edge = &base_path.edges[i - 1];
//                                 let shift_current = match vertices.last().unwrap() {
//                                     &Vertex::Surface(ref x) => x,
//                                     _ => panic!("Un-expected path for the shift mapping"), // If we have something else, panic!
//                                 };
//                                 // Check the visibility
//                                 if !scene.visible(&shift_current.its.p, &main_next.its.p) {
//                                     // Just return now the shift path is dead due to visibility
//                                     return None;
//                                 }

//                                 // Compute the new direction for evaluating the BSDF
//                                 let mut shift_d_out_global = main_next.its.p - shift_current.its.p;
//                                 let shift_distance = shift_d_out_global.magnitude();
//                                 shift_d_out_global /= shift_distance;
//                                 let shift_d_out_local =
//                                     shift_current.its.frame.to_local(shift_d_out_global);
//                                 // BSDF shift path
//                                 let shift_bsdf_value = shift_current.its.mesh.bsdf.eval(
//                                     &shift_current.its.uv,
//                                     &shift_current.its.wi,
//                                     &shift_d_out_local,
//                                 );
//                                 let shift_bsdf_pdf = match shift_current.its.mesh.bsdf.pdf(
//                                     &shift_current.its.uv,
//                                     &shift_current.its.wi,
//                                     &shift_d_out_local,
//                                 ) {
//                                     PDF::SolidAngle(x) => x,
//                                     _ => panic!("shift_bsdf_pdf is not in Solid angle"),
//                                 };

//                                 if shift_bsdf_pdf == 0.0 || shift_bsdf_value.is_zero() {
//                                     // Just return now the shift path as the rest of the vertex will be 0
//                                     return None;
//                                 }
//                                 // Compute the Jacobian value
//                                 let jacobian = (main_next.its.n_g.dot(-shift_d_out_global)
//                                     * main_next.its.dist.powi(2))
//                                     .abs()
//                                     / (main_next.its.n_g.dot(-main_edge.d)
//                                         * (shift_current.its.p - main_next.its.p).magnitude2())
//                                         .abs();
//                                 assert!(jacobian.is_finite());
//                                 assert!(jacobian >= 0.0);

//                                 Some((
//                                     Vertex::SurfaceShift(SurfaceVertexShift {
//                                         throughput: shift_current.throughput
//                                             * &(shift_bsdf_value * (jacobian / main_bsdf_pdf)),
//                                         pdf_ratio: pdf * (shift_bsdf_pdf * jacobian)
//                                             / main_bsdf_pdf,
//                                     }),
//                                     Edge {
//                                         dist: Some(shift_distance),
//                                         d: shift_d_out_global,
//                                     },
//                                 ))
//                             };

//                             // Update shift path
//                             match new_vertex {
//                                 None => return Some(Path { vertices, edges }),
//                                 Some((v, edge)) => {
//                                     vertices.push(v);
//                                     edges.push(edge);
//                                 }
//                             }

//                             // Change the state of the shift
//                             state = ShiftGeometricState::RecentlyConnected;
//                         }
//                         _ => panic!("Encounter wrong vertex type"),
//                     }
//                 }
//                 ShiftGeometricState::RecentlyConnected => {}
//                 ShiftGeometricState::Connected => {
//                     match &base_path.vertices.get(i) {
//                         &None => return Some(Path { vertices, edges }),
//                         &Some(&Vertex::Surface(ref main_next)) => {
//                             match &main_next.sampled_bsdf {
//                                 &Some(ref x) => {
//                                     let new_vertex = {
//                                         let shift_current = match vertices.last().unwrap() {
//                                             &Vertex::SurfaceShift(ref x) => x,
//                                             _ => panic!("Un-expected path for the shift mapping"), // If we have something else, panic!
//                                         };
//                                         Vertex::SurfaceShift(SurfaceVertexShift {
//                                             throughput: x.weight * shift_current.throughput,
//                                             pdf_ratio: shift_current.pdf_ratio, // No change here
//                                         })
//                                     };
//                                     // Just recopy the path
//                                     vertices.push(new_vertex);
//                                     edges.push(base_path.edges[i - 1].clone());
//                                 }
//                                 _ => {
//                                     // The main path is dead, stop doing the shift
//                                     // FIXME: Maybe one vertex will miss in this case
//                                     return Some(Path { vertices, edges });
//                                 }
//                             }
//                         }
//                         _ => panic!("Encounter wrong vertex type"),
//                     }
//                 }
//             }
//         }

//         Some(Path { vertices, edges })
//     }
// }
