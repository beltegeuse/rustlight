use cgmath::*;
use image::*;
use std::ops::{AddAssign, Mul, MulAssign, DivAssign, Div};
use std;

/// Pixel color representation
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Copy)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

pub trait Scale<T> {
    fn scale(&mut self, v: T);
}

impl Color {
    pub fn new(r: f32, g: f32, b: f32) -> Color {
        Color { r, g, b }
    }
    pub fn zero() -> Color {
        Color::new(0.0, 0.0, 0.0 )
    }
    pub fn one() -> Color {
        Color::new(1.0, 1.0, 1.0 )
    }
    pub fn value(v: f32) -> Color {
        Color::new(v, v, v)
    }

    pub fn is_zero(&self) -> bool {
        self.r == 0.0 && self.g == 0.0 && self.b == 0.0
    }
    pub fn to_rgba(&self) -> Rgba<u8> {
        Rgba::from_channels((self.r.min(1.0).powf(1.0 / 2.2) * 255.0) as u8,
                            (self.g.min(1.0).powf(1.0 / 2.2) * 255.0) as u8,
                            (self.b.min(1.0).powf(1.0 / 2.2) * 255.0) as u8,
                            255)
    }
    pub fn channel_max(&self) -> f32 {
        self.r.max(self.g.max(self.b))
    }
}
impl Default for Color {
    fn default() -> Self {
        Color::zero()
    }
}
impl Scale<f32> for Color {
    fn scale(&mut self, v: f32) {
        self.r *= v;
        self.g *= v;
        self.b *= v;
    }
}

/////////////// Operators
impl DivAssign<f32> for Color {
    fn div_assign(&mut self, other: f32) {
        self.r /= other;
        self.g /= other;
        self.b /= other;
    }
}

impl<'b> MulAssign<&'b Color> for Color {
    fn mul_assign(&mut self, other: &'b Color) {
        self.r *= other.r;
        self.g *= other.g;
        self.b *= other.b;
    }
}

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

impl Div<f32> for Color {
    fn div(self, other: f32) -> Color {
        assert!(other.is_finite());
        assert!(other != 0.0);
        Color {
            r: self.r / other,
            g: self.g / other,
            b: self.b / other,
        }
    }
    type Output = Self;
}

impl Mul<f32> for Color {
    fn mul(self, other: f32) -> Color {
        assert!(other.is_finite());
        Color {
            r: self.r * other,
            g: self.g * other,
            b: self.b * other,
        }
    }
    type Output = Self;
}

impl Mul<Color> for f32 {
    fn mul(self, other: Color) -> Color {
        Color {
            r: other.r * self,
            g: other.g * self,
            b: other.b * self,
        }
    }
    type Output = Color;
}

impl<'a> Mul<&'a Color> for f32 {
    fn mul(self, other: &'a Color) -> Color {
        Color {
            r: other.r * self,
            g: other.g * self,
            b: other.b * self,
        }
    }
    type Output = Color;
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

impl Mul<Color> for Color {
    fn mul(self, other: Color) -> Color {
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
    pub tnear : f32,
    pub tfar : f32,
}

impl Ray {
    pub fn new(o: Point3<f32>, d: Vector3<f32>) -> Ray {
        Ray {
            o,
            d,
            tnear: 0.0001, // Epsilon
            tfar: std::f32::MAX
        }
    }
}


