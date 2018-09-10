use cgmath::Point2;
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

pub mod path;
