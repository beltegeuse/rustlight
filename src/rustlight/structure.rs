use cgmath::*;
use image::*;
use std::ops::{AddAssign, Mul};

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

impl Mul<f32> for Color {
    fn mul(self, other: f32) -> Color {
        Color {
            r: self.r * other,
            g: self.g * other,
            b: self.b * other,
        }
    }
    type Output = Self;
}

impl Color {
    pub fn new(r: f32, g: f32, b: f32) -> Color {
        Color {
            r, g, b
        }
    }

    pub fn one(v: f32) -> Color {
        Color {
            r:v, g:v, b:v
        }
    }

    pub fn to_rgba(&self) -> Rgba<u8> {
        Rgba::from_channels((self.r * 255.0) as u8,
                            (self.g * 255.0) as u8,
                            (self.b * 255.0) as u8,
                            255)
    }

    pub fn mul(&mut self, v: f32) {
        self.r *= v;
        self.g *= v;
        self.b *= v;
    }
}

///////// Ray
pub struct Ray {
    pub o: Point3<f32>,
    pub d: Vector3<f32>,
}


