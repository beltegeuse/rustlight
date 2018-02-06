use cgmath::*;
use image::*;
use std::ops::{AddAssign};

//////// Color
#[derive(Clone, PartialEq, Debug)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl<'b> AddAssign<&'b Color> for Color {
    fn add_assign(&mut self, other: &'b Color) {
        self.r += other.r;
        self.g += other.g;
        self.b += other.b;
    }
}

impl Color {
    pub fn to_rgba(&self) -> Rgba<u8> {
        Rgba::from_channels((self.r * 255.0) as u8,
                            (self.g * 255.0) as u8,
                            (self.b * 255.0) as u8,
                            255)
    }
}

///////// Ray
pub struct Ray {
    pub o: Point3<f32>,
    pub d: Vector3<f32>,
}


