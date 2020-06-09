use crate::integrators::gradient::shiftmapping::{random_replay::RandomReplay, ShiftMapping};
use crate::integrators::{gradient::*, *};
use crate::paths::path::*;
use crate::paths::vertex::*;
use cgmath::Point2;

/// Path tracing system
/// This structure store the rendering options
/// That the user have given through the command line
pub struct IntegratorGradientPathTracing {
    pub max_depth: Option<u32>,
    pub recons: Box<dyn PoissonReconstruction + Sync>,
    pub min_survival: Option<f32>,
}
/// This structure is responsible to the graph generation
pub struct TechniqueGradientPathTracing {
    pub max_depth: Option<u32>,
    pub samplings: Vec<Box<dyn SamplingStrategy>>,
    pub img_pos: Point2<u32>,
}
impl Technique for TechniqueGradientPathTracing {
    fn init<'scene, 'emitter>(
        &mut self,
        path: &mut Path<'scene, 'emitter>,
        _accel: &'scene dyn Acceleration,
        scene: &'scene Scene,
        sampler: &mut dyn Sampler,
        _emitters: &'emitter EmitterSampler,
    ) -> Vec<(VertexID, Color)> {
        // Only generate a path from the sensor
        let root = Vertex::Sensor {
            uv: Point2::new(
                self.img_pos.x as f32 + sampler.next(),
                self.img_pos.y as f32 + sampler.next(),
            ),
            pos: scene.camera.position(),
            edge_in: None,
            edge_out: None,
        };

        return vec![(path.register_vertex(root), Color::one())];
    }

    fn expand(&self, _vertex: &Vertex, depth: u32) -> bool {
        self.max_depth.map_or(true, |max| depth < max)
    }

    fn strategies(&self, _vertex: &Vertex) -> &Vec<Box<dyn SamplingStrategy>> {
        &self.samplings
    }
}
impl TechniqueGradientPathTracing {
    pub fn evaluate<'scene, 'emitter>(
        &self,
        path: &Path<'scene, '_>,
        scene: &'scene Scene,
        emitters: &'emitter EmitterSampler,
        vertex_id: VertexID,
    ) -> Color {
        let mut l_i = Color::zero();
        match path.vertex(vertex_id) {
            Vertex::Surface { edge_out, .. } => {
                for edge_id in edge_out {
                    let edge = path.edge(*edge_id);
                    let contrib = edge.contribution(path);
                    if !contrib.is_zero() {
                        let weight = if let PDF::SolidAngle(v) = edge.pdf_direction {
                            let total: f32 = self
                                .strategies(path.vertex(vertex_id))
                                .iter()
                                .map(|s| {
                                    if let Some(v) =
                                        s.pdf(path, scene, emitters, vertex_id, *edge_id)
                                    {
                                        v
                                    } else {
                                        0.0
                                    }
                                })
                                .sum();
                            v / total
                        } else {
                            1.0
                        };
                        l_i += contrib * weight;
                    }

                    if let Some(vertex_next_id) = edge.vertices.1 {
                        l_i += edge.weight
                            * edge.rr_weight
                            * self.evaluate(path, scene, emitters, vertex_next_id);
                    }
                }
            }
            Vertex::Sensor { edge_out, .. } => {
                // Only one strategy where...
                let edge = path.edge(edge_out.unwrap());

                // Get the potential contribution
                let contrib = edge.contribution(path);
                if !contrib.is_zero() {
                    l_i += contrib;
                }

                // Do the reccursive call
                if let Some(vertex_next_id) = edge.vertices.1 {
                    l_i += edge.weight * self.evaluate(path, scene, emitters, vertex_next_id);
                }
            }
            _ => {}
        };
        l_i
    }
}
impl Integrator for IntegratorGradientPathTracing {}
impl IntegratorGradient for IntegratorGradientPathTracing {
    fn reconstruct(&self) -> &(dyn PoissonReconstruction + Sync) {
        self.recons.as_ref()
    }

    fn compute_gradients(&mut self, sampler: &mut dyn Sampler, accel: &dyn Acceleration, scene: &Scene) -> BufferCollection {
        let (nb_buffers, buffernames, mut image_blocks, ids) =
            generate_img_blocks_gradient(sampler, scene, self.recons.as_ref());

        let progress_bar = Mutex::new(ProgressBar::new(image_blocks.len() as u64));
        let pool = generate_pool(scene);
        pool.install(|| {
            image_blocks.par_iter_mut().for_each(|(info, im_block, sampler)| {
                let mut shiftmapping = RandomReplay::default();
                let emitters = scene.emitters_sampler();
                for ix in info.x_pos_off..im_block.size.x - info.x_size_off {
                    for iy in info.y_pos_off..im_block.size.y - info.y_size_off {
                        for n in 0..scene.nb_samples {
                            shiftmapping.clear();
                            let c = self.compute_pixel(
                                (ix + im_block.pos.x, iy + im_block.pos.y),
                                accel,
                                scene,
                                &emitters,
                                sampler.as_mut(),
                                &mut shiftmapping,
                            );
                            // Accumulate the values inside the buffer
                            let pos = Point2::new(ix, iy);
                            let offset_buffers = (n % nb_buffers) * 3; // 3 buffers are in multiple version
                            im_block.accumulate(
                                pos,
                                c.main,
                                &buffernames[ids.primal + offset_buffers],
                            );
                            im_block.accumulate(
                                pos,
                                c.very_direct,
                                &buffernames[ids.very_direct].to_owned(),
                            );
                            for i in 0..4 {
                                // primal reuse
                                let off = GRADIENT_ORDER[i];
                                let pos_off = Point2::new(ix as i32 + off.x, iy as i32 + off.y);
                                im_block.accumulate_safe(
                                    pos_off,
                                    c.radiances[i],
                                    &buffernames[ids.primal + offset_buffers],
                                );
                                // gradient
                                match GRADIENT_DIRECTION[i] {
                                    GradientDirection::X(v) => match v {
                                        1 => im_block.accumulate(
                                            pos,
                                            c.gradients[i],
                                            &buffernames[ids.gradient_x + offset_buffers],
                                        ),
                                        -1 => im_block.accumulate_safe(
                                            pos_off,
                                            c.gradients[i] * -1.0,
                                            &buffernames[ids.gradient_x + offset_buffers],
                                        ),
                                        _ => panic!("wrong displacement X"), // FIXME: Fix the enum
                                    },
                                    GradientDirection::Y(v) => match v {
                                        1 => im_block.accumulate(
                                            pos,
                                            c.gradients[i],
                                            &buffernames[ids.gradient_y + offset_buffers],
                                        ),
                                        -1 => im_block.accumulate_safe(
                                            pos_off,
                                            c.gradients[i] * -1.0,
                                            &buffernames[ids.gradient_y + offset_buffers],
                                        ),
                                        _ => panic!("wrong displacement Y"),
                                    },
                                }
                            }
                        }
                    }
                }
                im_block.scale(1.0 / (scene.nb_samples as f32));
                // Renormalize correctly the buffer informations
                for i in 0..nb_buffers {
                    let offset_buffers = i * 3; // 3 buffer that have multiple entries
                                                // 4 strategies as reuse primal
                    im_block.scale_buffer(
                        0.25 * nb_buffers as f32,
                        &buffernames[ids.primal + offset_buffers],
                    );
                    im_block.scale_buffer(
                        nb_buffers as f32,
                        &buffernames[ids.gradient_x + offset_buffers],
                    );
                    im_block.scale_buffer(
                        nb_buffers as f32,
                        &buffernames[ids.gradient_y + offset_buffers],
                    );
                }

                {
                    progress_bar.lock().unwrap().inc();
                }
            });
        });

        // Fill the image & do the reconstruct
        let mut image =
            BufferCollection::new(Point2::new(0, 0), *scene.camera.size(), &buffernames);
        for (_, im_block, _) in &image_blocks {
            image.accumulate_bitmap(im_block);
        }
        image
    }
}

impl IntegratorGradientPathTracing {
    fn compute_pixel<T: ShiftMapping>(
        &self,
        (ix, iy): (u32, u32),
        accel: &dyn Acceleration,
        scene: &Scene,
        emitters: &EmitterSampler,
        sampler: &mut dyn Sampler,
        shiftmapping: &mut T,
    ) -> ColorGradient {
        let mut path = Path::default();
        let mut samplings: Vec<Box<dyn SamplingStrategy>> = Vec::new();
        samplings.push(Box::new(DirectionalSamplingStrategy { from_sensor: true }));
        samplings.push(Box::new(LightSamplingStrategy {}));
        let mut technique = TechniqueGradientPathTracing {
            max_depth: None, // FIXME
            samplings,
            img_pos: Point2::new(0, 0), // FIXME
        };

        let (base_contrib, base_path) = shiftmapping.base(
            &mut path,
            &mut technique,
            Point2::new(ix, iy),
            accel,
            scene,
            emitters,
            sampler,
        );
        let weight_survival = if let Some(min_survival) = self.min_survival {
            // TODO: Change the 0.1 hard coded to a more meaningful value
            let prob_survival = (base_contrib.luminance() / 0.1).min(1.0).max(min_survival);
            if prob_survival == 1.0 || prob_survival >= sampler.next() {
                1.0 / prob_survival
            } else {
                0.0
            }
        } else {
            1.0
        };

        if weight_survival != 0.0 {
            let mut output = ColorGradient {
                very_direct: Color::zero(),
                main: Color::zero(),
                radiances: [Color::zero(); 4],
                gradients: [Color::zero(); 4],
            };

            GRADIENT_ORDER.iter().enumerate().for_each(|(i, off)| {
                let pix = Point2::new(ix as i32 + off.x, iy as i32 + off.y);
                if pix.x < 0
                    || pix.x > scene.camera.size().x as i32
                    || pix.y < 0
                    || pix.y > scene.camera.size().y as i32
                {
                    // Do nothing
                } else {
                    // Change the pixel for the sampling technique
                    // and reset the sampler
                    let shift_value = shiftmapping.shift(
                        &mut path,
                        &mut technique,
                        Point2::new(pix.x as u32, pix.y as u32),
                        accel,
                        scene,
                        emitters,
                        sampler,
                        base_path,
                    );
                    output.main += shift_value.base * weight_survival;
                    output.radiances[i] = shift_value.offset * weight_survival;
                    output.gradients[i] = shift_value.gradient * weight_survival;
                }
            });
            output
        } else {
            ColorGradient::default()
        }
    }
}
