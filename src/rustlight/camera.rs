use cgmath::*;
use rustlight::structure::{Ray};
use std::f32;

pub struct Camera {
    pub pos: Point3<f32>,
    pub dir: Vector3<f32>,

    // Internally
    dir_top_left: Vector3<f32>,
    screen_du: Vector3<f32>,
    screen_dv: Vector3<f32>,
    pub img: Vector2<u32>,
}

impl Camera {
    pub fn new(pos: Point3<f32>, dir: Vector3<f32>, up: Vector3<f32>, fov: f32, img: Vector2<u32>) -> Camera {
        let dz = dir.normalize();
        let dx = -dz.cross(up).normalize();
        let dy = dx.cross(dz).normalize();
        let dim_y = 2.0 * f32::tan((fov / 2.0) * f32::consts::PI / 180.0);
        let aspect_ratio = img.x as f32 / img.y as f32;
        let dim_x = dim_y * aspect_ratio;
        let screen_du = dx * dim_x;
        let screen_dv = dy * dim_y;
        let dir_top_left = dz - 0.5 * screen_du - 0.5 * screen_dv;
        Camera { pos: pos, dir: dz, dir_top_left: dir_top_left, screen_du: screen_du,
            screen_dv: screen_dv, img: img }
    }

    pub fn look_at(pos: Point3<f32>, at: Point3<f32>, up: Vector3<f32>, fov: f32, img: Vector2<u32>) -> Camera {
        let dir = at - pos;
        Camera::new(pos, dir, up, fov, img)
    }

    /// Compute the ray direction going through the pixel passed
    pub fn generate(&self, px: (f32, f32)) -> Ray {
        let d = (self.dir_top_left + px.0 / (self.img.x as f32) * self.screen_du
            + px.1 / (self.img.y as f32) * self.screen_dv).normalize();

        Ray {
            o: self.pos.clone(),
            d: d,
        }
    }
}
