use integrators::gradient::*;

pub struct WeightedPoissonReconstruction {
    pub iterations: usize,
}
impl PoissonReconstruction for WeightedPoissonReconstruction {
    fn need_variance_estimates(&self) -> Option<usize> {
        Some(2)
    }

    fn reconstruct(&self, scene: &Scene, est: &Bitmap) -> Bitmap {
        // Reconstruction (image-space covariate, uniform reconstruction)
        let img_size = est.size;
        let buffernames = vec!["recons".to_string()];
        let mut current = Bitmap::new(Point2::new(0, 0), img_size.clone(), &buffernames);
        let mut image_blocks = generate_img_blocks(scene, &buffernames);

        // Define names of buffers so we do not need to reallocate them
        let primal_name = String::from("primal_0");
        let recons_name = String::from("recons");
        let gradient_x_name = String::from("gradient_x_0");
        let gradient_y_name = String::from("gradient_y_0");
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
        let real_primal_name = String::from("primal");
        let mut image: Bitmap = Bitmap::new(
            Point2::new(0, 0),
            img_size.clone(),
            &vec![real_primal_name.clone()],
        );
        for x in 0..img_size.x {
            for y in 0..img_size.y {
                let pos = Point2::new(x, y);
                let pix_value = current.get(pos, &recons_name).clone()
                    + est.get(pos, &very_direct_name).clone();
                image.accumulate(pos, pix_value, &real_primal_name);
            }
        }
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
        for x in 0..img_size.x {
            for y in 0..img_size.y {
                let pos = Point2::new(x, y);
                let pix_value = current.get(pos, &recons_name).clone()
                    + est.get(pos, &very_direct_name).clone();
                image.accumulate(pos, pix_value, &primal_name);
            }
        }
        image
    }
}
