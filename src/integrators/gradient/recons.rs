use cgmath::Vector2;
use integrators::gradient::*;
use rayon::prelude::*;
use Scale;

pub struct BaggingPoissonReconstruction {
    pub iterations: usize,
    pub nb_buffers: usize,
}
impl PoissonReconstruction for BaggingPoissonReconstruction {
    fn need_variance_estimates(&self) -> Option<usize> {
        Some(self.nb_buffers)
    }

    fn reconstruct(&self, scene: &Scene, est: &Bitmap) -> Bitmap {
        let img_size = est.size;

        // Generate several reconstruction and average it
        // For now, the number of reconstruction is equal to the number of buffers -1
        if self.nb_buffers < 2 {
            panic!("Impossible to do bagging with less than two buffers");
        }

        let mut image_recons =Bitmap::new(
            Point2::new(0, 0),
            img_size.clone(),
            &Vec::new());
        let mut buffernames = Vec::new();
        for n_recons in 0..self.nb_buffers {
            // Construct the buffer id
            // by excluding one bucket
            let mut buffer_id = Vec::new();
            for i in 0..self.nb_buffers {
                if i == n_recons {
                    continue;
                }
                buffer_id.push(i);
            }

            // Do the reconstruction
            let weighted_recons = WeightedPoissonReconstruction::new(self.iterations).restrict_buffers(buffer_id);
            info!("Reconstruction {} / {}", n_recons+1, self.nb_buffers);
            let image_res = weighted_recons.reconstruct(scene, est);
            let image_name = format!("primal_{}",n_recons);
            image_recons.register(image_name.clone());
            image_recons.accumulate_bitmap_buffer(&image_res, &"primal".to_string(), &image_name);
            buffernames.push(image_name);
        }

        // Average the different results
        let mut image_avg = Bitmap::new(
            Point2::new(0, 0),
            img_size.clone(),
            &Vec::new());
        image_avg.register_mean_variance(&"primal".to_string(), &image_recons, &buffernames);
        //image_avg.dump_all(scene.output_img_path.clone());
        image_avg.rename(&"primal_mean".to_string(), &"primal".to_string());
        image_avg
    }
}

pub struct WeightedPoissonReconstruction {
    pub iterations: usize,
    buffers_id: Option<Vec<usize>>, //< Only to select few buffers for the rendering
}
impl WeightedPoissonReconstruction {
    pub fn new(iterations: usize) -> WeightedPoissonReconstruction {
        WeightedPoissonReconstruction {
            iterations,
            buffers_id: None,
        }
    }

    pub fn restrict_buffers(mut self, buffer_id: Vec<usize>) -> WeightedPoissonReconstruction {
        self.buffers_id = Some(buffer_id);
        self
    }

    fn generate_average_variance_bitmap(&self, est: &Bitmap, img_size: &Vector2<u32>) -> Bitmap {
        let mut averaged_variance = Bitmap::new(Point2::new(0, 0), img_size.clone(), &Vec::new());
        let buffernames = vec![
            String::from("primal"),
            String::from("gradient_x"),
            String::from("gradient_y"),
        ];
        for buffer in buffernames {
            let mut selected_names = match self.buffers_id.as_ref() {
                None => {
                    let nb_buffers = self.need_variance_estimates().unwrap();
                    (0..nb_buffers)
                        .into_iter()
                        .map(|i| format!("{}_{}", buffer, i))
                        .collect()
                }
                Some(ref v) => v.iter().map(|i| format!("{}_{}", buffer, i)).collect(),
            };
            averaged_variance.register_mean_variance(&buffer, est, &selected_names);
        }
        averaged_variance
    }
}

impl PoissonReconstruction for WeightedPoissonReconstruction {
    fn need_variance_estimates(&self) -> Option<usize> {
        match self.buffers_id.as_ref() {
            None => Some(2),
            Some(ref v) => Some(v.len()),
        }
    }

    fn reconstruct(&self, scene: &Scene, est: &Bitmap) -> Bitmap {
        let inv_or_1 = |v| if v == 0.0 { 1.0 } else { 1.0 / v };

        // Reconstruction (image-space covariate, uniform reconstruction)
        let img_size = est.size;

        // Average the different buffers
        let averaged_variance = self.generate_average_variance_bitmap(est, &img_size);

        // Define names of buffers so we do not need to reallocate them
        let primal_name = String::from("primal_mean");
        let recons_name = String::from("recons");
        let gradient_x_name = String::from("gradient_x_mean");
        let gradient_y_name = String::from("gradient_y_mean");
        let very_direct_name = String::from("very_direct");

        // And variances
        let gradient_x_variance_name = String::from("gradient_x_variance");
        let gradient_y_variance_name = String::from("gradient_y_variance");
        let primal_variance_name = String::from("primal_variance");

        // 1) Init
        let buffernames = vec![recons_name.clone()];
        let mut current = Bitmap::new(Point2::new(0, 0), img_size.clone(), &buffernames);
        current.accumulate_bitmap_buffer(&averaged_variance, &primal_name, &recons_name);

        // Generate the buffer names
        let mut image_blocks = generate_img_blocks(scene, &buffernames);
        let pool = generate_pool(scene);
        pool.install(|| {
            for iter in 0..self.iterations {
                image_blocks.par_iter_mut().for_each(|im_block| {
                    im_block.reset();
                    for local_y in 0..im_block.size.y {
                        for local_x in 0..im_block.size.x {
                            let (x, y) = (local_x + im_block.pos.x, local_y + im_block.pos.y);
                            let pos = Point2::new(x, y);

                            // Compute variance inside the current pixel
                            let coeff_var_red =
                                1.0 / (0.01 + 1.0 + 4.0 * 0.5_f32.powf(iter as f32));
                            let var_pos = averaged_variance
                                .get(pos, &primal_variance_name)
                                .channel_max()
                                * coeff_var_red;
                            let curr_weight = inv_or_1(var_pos);
                            let mut c = current.get(pos, &recons_name).clone() * curr_weight;
                            let mut w = curr_weight;

                            if x > 0 {
                                let pos_off = Point2::new(x - 1, y);
                                let curr_weight = inv_or_1(
                                    var_pos
                                        + averaged_variance
                                            .get(pos_off, &gradient_x_variance_name)
                                            .channel_max(),
                                );
                                c += (current.get(pos_off, &recons_name).clone()
                                    + averaged_variance.get(pos_off, &gradient_x_name).clone())
                                    * curr_weight;
                                w += curr_weight;
                            }
                            if x < img_size.x - 1 {
                                let pos_off = Point2::new(x + 1, y);
                                let curr_weight = inv_or_1(
                                    var_pos
                                        + averaged_variance
                                            .get(pos, &gradient_x_variance_name)
                                            .channel_max(),
                                );
                                c += (current.get(pos_off, &recons_name).clone()
                                    - averaged_variance.get(pos, &gradient_x_name).clone())
                                    * curr_weight;
                                w += curr_weight;
                            }
                            if y > 0 {
                                let pos_off = Point2::new(x, y - 1);
                                let curr_weight = inv_or_1(
                                    var_pos
                                        + averaged_variance
                                            .get(pos_off, &gradient_y_variance_name)
                                            .channel_max(),
                                );
                                c += (current.get(pos_off, &recons_name).clone()
                                    + averaged_variance.get(pos_off, &gradient_y_name).clone())
                                    * curr_weight;
                                w += curr_weight;
                            }
                            if y < img_size.y - 1 {
                                let pos_off = Point2::new(x, y + 1);
                                let curr_weight = inv_or_1(
                                    var_pos
                                        + averaged_variance
                                            .get(pos, &gradient_y_variance_name)
                                            .channel_max(),
                                );
                                c += (current.get(pos_off, &recons_name).clone()
                                    - averaged_variance.get(pos, &gradient_y_name).clone())
                                    * curr_weight;
                                w += curr_weight;
                            }
                            c.scale(1.0 / w);
                            im_block.accumulate(Point2::new(local_x, local_y), c, &recons_name);
                        }
                    }
                });
                // Collect the data
                current.reset();
                for im_block in &image_blocks {
                    current.accumulate_bitmap(im_block);
                }
            }
        });

        // Export the reconstruction
        let real_primal_name = String::from("primal");
        let mut image: Bitmap = Bitmap::new(
            Point2::new(0, 0),
            img_size.clone(),
            &vec![real_primal_name.clone()],
        );
        image.accumulate_bitmap_buffer(&current, &recons_name, &real_primal_name);
        image.accumulate_bitmap_buffer(&est, &very_direct_name, &real_primal_name);
        image
    }
}

pub struct UniformPoissonReconstruction {
    pub iterations: usize,
}
impl PoissonReconstruction for UniformPoissonReconstruction {
    fn need_variance_estimates(&self) -> Option<usize> {
        None
    }

    fn reconstruct(&self, scene: &Scene, est: &Bitmap) -> Bitmap {
        // Reconstruction (image-space covariate, uniform reconstruction)
        let img_size = est.size;
        let buffernames = vec!["recons".to_string()];
        let mut current = Bitmap::new(Point2::new(0, 0), img_size.clone(), &buffernames);
        let mut image_blocks = generate_img_blocks(scene, &buffernames);

        // Define names of buffers so we do not need to reallocate them
        let primal_name = String::from("primal");
        let recons_name = String::from("recons");
        let gradient_x_name = String::from("gradient_x");
        let gradient_y_name = String::from("gradient_y");
        let very_direct_name = String::from("very_direct");

        // 1) Init
        for y in 0..img_size.y {
            for x in 0..img_size.x {
                let pos = Point2::new(x, y);
                current.accumulate(pos, *est.get(pos, &primal_name), &recons_name);
            }
        }

        let pool = generate_pool(scene);
        // 2) Iterations
        pool.install(|| {
            for _iter in 0..self.iterations {
                image_blocks.par_iter_mut().for_each(|im_block| {
                    im_block.reset();
                    for local_y in 0..im_block.size.y {
                        for local_x in 0..im_block.size.x {
                            let (x, y) = (local_x + im_block.pos.x, local_y + im_block.pos.y);
                            let pos = Point2::new(x, y);
                            let mut c = current.get(pos, &recons_name).clone();
                            let mut w = 1.0;
                            if x > 0 {
                                let pos_off = Point2::new(x - 1, y);
                                c += current.get(pos_off, &recons_name).clone()
                                    + est.get(pos_off, &gradient_x_name).clone();
                                w += 1.0;
                            }
                            if x < img_size.x - 1 {
                                let pos_off = Point2::new(x + 1, y);
                                c += current.get(pos_off, &recons_name).clone()
                                    - est.get(pos, &gradient_x_name).clone();
                                w += 1.0;
                            }
                            if y > 0 {
                                let pos_off = Point2::new(x, y - 1);
                                c += current.get(pos_off, &recons_name).clone()
                                    + est.get(pos_off, &gradient_y_name).clone();
                                w += 1.0;
                            }
                            if y < img_size.y - 1 {
                                let pos_off = Point2::new(x, y + 1);
                                c += current.get(pos_off, &recons_name).clone()
                                    - est.get(pos, &gradient_y_name).clone();
                                w += 1.0;
                            }
                            c.scale(1.0 / w);
                            im_block.accumulate(Point2::new(local_x, local_y), c, &recons_name);
                        }
                    }
                });
                // Collect the data
                current.reset();
                for im_block in &image_blocks {
                    current.accumulate_bitmap(im_block);
                }
            }
        });
        // Export the reconstruction
        let mut image: Bitmap = Bitmap::new(
            Point2::new(0, 0),
            img_size.clone(),
            &vec![String::from("primal")],
        );
        image.accumulate_bitmap_buffer(&current, &recons_name, &primal_name);
        image.accumulate_bitmap_buffer(&est, &very_direct_name, &primal_name);
        image
    }
}
