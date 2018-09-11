use cgmath::Point2;
use integrators::{generate_img_blocks, generate_pool, Bitmap, Integrator};
use rayon::prelude::*;
use scene::Scene;
use std::time::Instant;
use structure::Color;
use Scale;

#[derive(Clone, Debug, Copy)]
pub struct ColorGradient {
    pub very_direct: Color,
    pub main: Color,
    pub radiances: [Color; 4],
    pub gradients: [Color; 4],
}
impl Default for ColorGradient {
    fn default() -> Self {
        ColorGradient {
            very_direct: Color::zero(),
            main: Color::zero(),
            radiances: [Color::zero(); 4],
            gradients: [Color::zero(); 4],
        }
    }
}

pub enum GradientDirection {
    X(i32),
    Y(i32),
}

pub static GRADIENT_ORDER: [Point2<i32>; 4] = [
    Point2 { x: 0, y: 1 },
    Point2 { x: 0, y: -1 },
    Point2 { x: 1, y: 0 },
    Point2 { x: -1, y: 0 },
];
pub static GRADIENT_DIRECTION: [GradientDirection; 4] = [
    GradientDirection::Y(1),
    GradientDirection::Y(-1),
    GradientDirection::X(1),
    GradientDirection::X(-1),
];

pub trait IntegratorGradient: Integrator {
    fn compute_gradients(&mut self, scene: &Scene) -> Bitmap;
    fn reconstruct(&self) -> &Box<PoissonReconstruction + Sync>;

    fn compute(&mut self, scene: &Scene) -> Bitmap {
        // Rendering the gradient informations
        info!("Rendering...");
        let start = Instant::now();
        let image = self.compute_gradients(scene);
        let elapsed = start.elapsed();
        info!(
            "Rendering Elapsed: {} ms",
            (elapsed.as_secs() * 1_000) + (elapsed.subsec_nanos() / 1_000_000) as u64
        );

        // Reconstruct the image
        info!("Reconstruction...");
        let start = Instant::now();
        let image = self.reconstruct().reconstruct(scene, &image);
        let elapsed = start.elapsed();
        info!(
            "Reconstruction Elapsed: {} ms",
            (elapsed.as_secs() * 1_000) + (elapsed.subsec_nanos() / 1_000_000) as u64
        );

        image
    }
}

pub trait PoissonReconstruction {
    fn reconstruct(&self, scene: &Scene, est: &Bitmap) -> Bitmap;
    fn need_variance_estimates(&self) -> bool;
}
pub struct UniformPoissonReconstruction {
    pub iterations: usize,
}
impl PoissonReconstruction for UniformPoissonReconstruction {
    fn need_variance_estimates(&self) -> bool {
        false
    }

    fn reconstruct(&self, scene: &Scene, est: &Bitmap) -> Bitmap {
        // Reconstruction (image-space covariate, uniform reconstruction)
        let img_size = est.size;
        let buffernames = vec!["recons"];
        let mut current = Bitmap::new(Point2::new(0, 0), img_size.clone(), &buffernames);
        let mut image_blocks = generate_img_blocks(scene, &buffernames);

        // 1) Init
        for y in 0..img_size.y {
            for x in 0..img_size.x {
                let pos = Point2::new(x, y);
                current.accumulate(pos, *est.get(pos, "primal"), "recons");
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
                            let mut c = current.get(pos, "recons").clone();
                            let mut w = 1.0;
                            if x > 0 {
                                let pos_off = Point2::new(x - 1, y);
                                c += current.get(pos_off, "recons").clone()
                                    + est.get(pos_off, "gradient_x").clone();
                                w += 1.0;
                            }
                            if x < img_size.x - 1 {
                                let pos_off = Point2::new(x + 1, y);
                                c += current.get(pos_off, "recons").clone()
                                    - est.get(pos, "gradient_x").clone();
                                w += 1.0;
                            }
                            if y > 0 {
                                let pos_off = Point2::new(x, y - 1);
                                c += current.get(pos_off, "recons").clone()
                                    + est.get(pos_off, "gradient_y").clone();
                                w += 1.0;
                            }
                            if y < img_size.y - 1 {
                                let pos_off = Point2::new(x, y + 1);
                                c += current.get(pos_off, "recons").clone()
                                    - est.get(pos, "gradient_y").clone();
                                w += 1.0;
                            }
                            c.scale(1.0 / w);
                            im_block.accumulate(Point2::new(local_x, local_y), c, "recons");
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
        let mut image: Bitmap = Bitmap::new(Point2::new(0, 0), img_size.clone(), &vec!["primal"]);
        for x in 0..img_size.x {
            for y in 0..img_size.y {
                let pos = Point2::new(x, y);
                let pix_value =
                    current.get(pos, "recons").clone() + est.get(pos, "very_direct").clone();
                image.accumulate(pos, pix_value, "primal");
            }
        }
        image
    }
}

pub mod avg;
pub mod path;
