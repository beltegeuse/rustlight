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

impl Ray {
    pub fn generate(impos: Point2<u32>, res: &Point2<u32>, fov : f32) -> Ray {
        assert!(res.x > res.y);
        let fov_adjustment = (fov.to_radians() / 2.0).tan();
        let aspect_ratio = (res.x as f32) / (res.y as f32);

        let sensor_x = (((impos.x as f32 + 0.5) / res.x as f32) * 2.0 - 1.0) * aspect_ratio * fov_adjustment;
        let sensor_y = (1.0 - ((impos.y as f32 + 0.5) / res.y as f32) * 2.0) * fov_adjustment;

        Ray {
            o: Point3 { x: 0.0, y: 0.0, z: 0.0 },
            d: Vector3 { x: sensor_x, y: sensor_y, z: -1.0 }.normalize(),
        }
    }
}


