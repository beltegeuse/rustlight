use cgmath::Point2;
use integrators::{generate_img_blocks, generate_pool, Bitmap, Integrator};
use scene::Scene;
use std::time::Instant;
use structure::Color;

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
        info!("Rendering Elapsed: {:?}", elapsed,);

        // Reconstruct the image
        info!("Reconstruction...");
        let start = Instant::now();
        let image = self.reconstruct().reconstruct(scene, &image);
        let elapsed = start.elapsed();
        info!("Reconstruction Elapsed: {:?}", elapsed,);

        image
    }
}

pub trait PoissonReconstruction {
    fn reconstruct(&self, scene: &Scene, est: &Bitmap) -> Bitmap;
    fn need_variance_estimates(&self) -> Option<usize>;
}

pub mod avg;
pub mod path;
pub mod recons;
