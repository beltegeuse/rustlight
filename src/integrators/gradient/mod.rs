use crate::integrators::{generate_img_blocks, generate_pool, BufferCollection, Integrator};
use crate::scene::Scene;
use crate::structure::Color;
use crate::tools::StepRangeInt;
use cgmath::{Point2, Vector2};
use std::cmp;
use std::time::Instant;

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
    fn compute_gradients(&mut self, scene: &Scene) -> BufferCollection;
    fn reconstruct(&self) -> &Box<PoissonReconstruction + Sync>;

    fn compute(&mut self, scene: &Scene) -> BufferCollection {
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
    fn reconstruct(&self, scene: &Scene, est: &BufferCollection) -> BufferCollection;
    fn need_variance_estimates(&self) -> Option<usize>;
}

// Compare to path tracing, make the block a bit bigger
// so that we can store all the path contribution
pub struct BlockInfoGradient {
    pub x_pos_off: u32,
    pub y_pos_off: u32,
    pub x_size_off: u32,
    pub y_size_off: u32,
}
pub struct BufferIDGradient {
    pub very_direct: usize,
    pub primal: usize,
    pub gradient_x: usize,
    pub gradient_y: usize,
}
pub fn generate_img_blocks_gradient(
    scene: &Scene,
    recons: &Box<PoissonReconstruction + Sync>,
) -> (
    usize,
    Vec<String>,
    Vec<(BlockInfoGradient, BufferCollection)>,
    BufferIDGradient,
) {
    // The buffers names are always:
    // ["very_direct", ("primal", "gradient_x", "gradient_y")+]
    let (nb_buffers, buffernames) = if let Some(number_buffers) = recons.need_variance_estimates() {
        let mut buffernames = Vec::new();
        buffernames.reserve((3 * number_buffers) + 1);
        buffernames.push(String::from("very_direct"));
        for i in 0..number_buffers {
            buffernames.push(format!("primal_{}", i));
            buffernames.push(format!("gradient_x_{}", i));
            buffernames.push(format!("gradient_y_{}", i));
        }
        (number_buffers, buffernames)
    } else {
        (
            1,
            vec![
                String::from("very_direct"),
                String::from("primal"),
                String::from("gradient_x"),
                String::from("gradient_y"),
            ],
        )
    };

    let mut image_blocks = Vec::new();
    for ix in StepRangeInt::new(0, scene.camera.size().x as usize, 16) {
        for iy in StepRangeInt::new(0, scene.camera.size().y as usize, 16) {
            let pos_off = Point2 {
                x: cmp::max(0, ix as i32 - 1) as u32,
                y: cmp::max(0, iy as i32 - 1) as u32,
            };
            let desired_size = Vector2 {
                x: 16 + if ix == 0 { 1 } else { 2 },
                y: 16 + if iy == 0 { 1 } else { 2 },
            };
            let max_size = Vector2 {
                x: (scene.camera.size().x - pos_off.x) as u32,
                y: (scene.camera.size().y - pos_off.y) as u32,
            };
            let mut block = BufferCollection::new(
                pos_off,
                Vector2 {
                    x: cmp::min(desired_size.x, max_size.x),
                    y: cmp::min(desired_size.y, max_size.y),
                },
                &buffernames,
            );
            let info = BlockInfoGradient {
                x_pos_off: if ix == 0 { 0 } else { 1 },
                y_pos_off: if iy == 0 { 0 } else { 1 },
                x_size_off: if desired_size.x <= max_size.x { 1 } else { 0 },
                y_size_off: if desired_size.y <= max_size.y { 1 } else { 0 },
            };
            image_blocks.push((info, block));
        }
    }
    (
        nb_buffers,
        buffernames,
        image_blocks,
        BufferIDGradient {
            very_direct: 0,
            primal: 1,
            gradient_x: 2,
            gradient_y: 3,
        },
    )
}

pub mod avg;
pub mod explicit;
pub mod path;
pub mod recons;
pub mod shiftmapping;
