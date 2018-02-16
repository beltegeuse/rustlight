use cgmath::*;
use image::*;
use std::ops::{AddAssign, Mul};

/// Pixel color representation
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
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

    pub fn is_zero(&self) -> bool {
        self.r == 0.0 && self.g == 0.0 && self.b == 0.0
    }

    pub fn to_rgba(&self) -> Rgba<u8> {
        Rgba::from_channels((self.r * 255.0).min(255.0) as u8,
                            (self.g * 255.0).min(255.0) as u8,
                            (self.b * 255.0).min(255.0) as u8,
                            255)
    }

    pub fn mul(&mut self, v: f32) {
        self.r *= v;
        self.g *= v;
        self.b *= v;
    }
}


/////////////// Operators
impl<'b> AddAssign<&'b Color> for Color {
    fn add_assign(&mut self, other: &'b Color) {
        self.r += other.r;
        self.g += other.g;
        self.b += other.b;
    }
}

impl AddAssign<Color> for Color {
    fn add_assign(&mut self, other: Color) {
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

impl<'a> Mul<&'a Color> for Color {
    fn mul(self, other: &'a Color) -> Color {
        Color {
            r: self.r * other.r,
            g: self.g * other.g,
            b: self.b * other.b,
        }
    }
    type Output = Self;
}

// FIXME: Evaluate if we keep it or not
// FIXME: If we keep it, add tfar, tmin ...
/// Ray representation
pub struct Ray {
    pub o: Point3<f32>,
    pub d: Vector3<f32>,
}

impl Ray {
    pub fn new(o: Point3<f32>, d: Vector3<f32>) -> Ray {
        Ray {
            o,
            d,
        }
    }
}


