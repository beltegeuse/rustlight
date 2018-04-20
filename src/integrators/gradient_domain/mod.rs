use cgmath::*;
use structure::*;
use std::ops::AddAssign;
use Scale;

#[derive(Clone, Debug, Copy)]
pub struct ColorGradient {
    pub very_direct: Color,
    pub main: Color,
    pub radiances: [Color; 4],
    pub gradients: [Color; 4],
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

impl AddAssign<ColorGradient> for ColorGradient {
    fn add_assign(&mut self, other: ColorGradient) {
        self.very_direct += other.very_direct;
        self.main += other.main;
        for i in 0..self.gradients.len() {
            self.radiances[i] += other.radiances[i];
            self.gradients[i] += other.gradients[i];
        }
    }
}

impl Scale<f32> for ColorGradient {
    fn scale(&mut self, v: f32) {
        self.very_direct.scale(v);
        self.main.scale(v);
        for i in 0..self.gradients.len() {
            self.radiances[i].scale(v);
            self.gradients[i].scale(v);
        }
    }
}

pub mod gradient_path;
pub mod gradient_path_explicit;