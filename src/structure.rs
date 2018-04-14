use cgmath::*;
use constants;
use embree_rs;
use geometry::Mesh;
use image::*;
use math::Frame;
use Scale;
use std;
use std::ops::*;
use std::sync::Arc;

/// PDF represented into different spaces
#[derive(Clone)]
pub enum PDF {
    SolidAngle(f32),
    Area(f32),
    Discrete(f32),
}

impl PDF {
    pub fn is_zero(&self) -> bool {
        match self {
            &PDF::Discrete(v) | &PDF::SolidAngle(v) | &PDF::Area(v) => (v == 0.0),
        }
    }

    pub fn value(&self) -> f32 {
        match self {
            &PDF::Discrete(v) | &PDF::SolidAngle(v) | &PDF::Area(v) => v,
        }
    }
}

/// Pixel color representation
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Copy)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Color {
    pub fn new(r: f32, g: f32, b: f32) -> Color {
        Color { r, g, b }
    }
    pub fn zero() -> Color {
        Color::new(0.0, 0.0, 0.0)
    }
    pub fn one() -> Color {
        Color::new(1.0, 1.0, 1.0)
    }
    pub fn value(v: f32) -> Color {
        Color::new(v, v, v)
    }

    pub fn is_zero(&self) -> bool {
        self.r == 0.0 && self.g == 0.0 && self.b == 0.0
    }
    pub fn to_rgba(&self) -> Rgba<u8> {
        Rgba::from_channels(
            (self.r.min(1.0).powf(1.0 / 2.2) * 255.0) as u8,
            (self.g.min(1.0).powf(1.0 / 2.2) * 255.0) as u8,
            (self.b.min(1.0).powf(1.0 / 2.2) * 255.0) as u8,
            255,
        )
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
    type Output = Self;
    fn div(self, other: f32) -> Color {
        assert!(other.is_finite());
        assert_ne!(other, 0.0);
        Color {
            r: self.r / other,
            g: self.g / other,
            b: self.b / other,
        }
    }
}

impl Mul<f32> for Color {
    type Output = Self;
    fn mul(self, other: f32) -> Color {
        //assert!(other.is_finite());
        if other.is_finite() {
            Color {
                r: self.r * other,
                g: self.g * other,
                b: self.b * other,
            }
        } else {
            Color::zero()
        }
    }
}

impl Mul<Color> for f32 {
    type Output = Color;
    fn mul(self, other: Color) -> Color {
        Color {
            r: other.r * self,
            g: other.g * self,
            b: other.b * self,
        }
    }
}

impl<'a> Mul<&'a Color> for f32 {
    type Output = Color;
    fn mul(self, other: &'a Color) -> Color {
        Color {
            r: other.r * self,
            g: other.g * self,
            b: other.b * self,
        }
    }
}

impl<'a> Mul<&'a Color> for Color {
    type Output = Self;
    fn mul(self, other: &'a Color) -> Color {
        Color {
            r: self.r * other.r,
            g: self.g * other.g,
            b: self.b * other.b,
        }
    }
}

impl Mul<Color> for Color {
    type Output = Self;
    fn mul(self, other: Color) -> Color {
        Color {
            r: self.r * other.r,
            g: self.g * other.g,
            b: self.b * other.b,
        }
    }
}

impl Sub<Color> for Color {
    type Output = Self;
    fn sub(self, other: Color) -> Color {
        Color {
            r: self.r - other.r,
            g: self.g - other.g,
            b: self.b - other.b,
        }
    }
}

impl Add<Color> for Color {
    type Output = Self;
    fn add(self, other: Color) -> Color {
        Color {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
        }
    }
}

impl<'a> Add<&'a Color> for Color {
    type Output = Self;
    fn add(self, other: &'a Color) -> Color {
        Color {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
        }
    }
}

/// Ray representation
pub struct Ray {
    pub o: Point3<f32>,
    pub d: Vector3<f32>,
    pub tnear: f32,
    pub tfar: f32,
}

impl Ray {
    pub fn new(o: Point3<f32>, d: Vector3<f32>) -> Ray {
        Ray {
            o,
            d,
            tnear: constants::EPSILON,
            tfar: std::f32::MAX,
        }
    }

    pub fn to_embree(&self) -> embree_rs::ray::Ray {
        embree_rs::ray::Ray::new(&self.o, &self.d, self.tnear, self.tfar)
    }
}

#[derive(Clone)]
pub struct Intersection<'a> {
    /// Intersection distance
    pub dist: f32,
    /// Geometry normal
    pub n_g: Vector3<f32>,
    /// Shading normal
    pub n_s: Vector3<f32>,
    /// Intersection point
    pub p: Point3<f32>,
    /// Textures coordinates
    pub uv: Option<Vector2<f32>>,
    /// Mesh which we have intersected
    pub mesh: &'a Arc<Mesh>,
    /// Frame from the intersection point
    pub frame: Frame,
    /// Incomming direction in the local coordinates
    pub wi: Vector3<f32>,
}

impl<'a> Intersection<'a> {
    pub fn new(
        embree_its: embree_rs::ray::Intersection,
        d: Vector3<f32>,
        mesh: &'a Arc<Mesh>,
    ) -> Intersection<'a> {
        let frame = Frame::new(embree_its.n_s);
        let wi = frame.to_local(d);
        Intersection {
            dist: embree_its.t,
            n_g: embree_its.n_g,
            n_s: embree_its.n_s,
            p: embree_its.p,
            uv: embree_its.uv,
            mesh,
            frame,
            wi,
        }
    }

    pub fn cos_theta(&self) -> f32 {
        self.wi.z
    }
}
