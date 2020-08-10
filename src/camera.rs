use crate::structure::{Color, Ray};
use cgmath::*;
use std::f32;

pub struct Camera {
    pub img: Vector2<u32>,
    // Internally
    camera_to_sample: Matrix4<f32>,
    sample_to_camera: Matrix4<f32>,
    to_world: Matrix4<f32>,
    to_local: Matrix4<f32>,
    // image rect
    image_rect_min: Point2<f32>,
    image_rect_max: Point2<f32>,
}

pub enum Fov {
    Y(f32),
    X(f32),
}
impl Fov {
    pub fn value(&self) -> f32 {
        match self {
            Fov::Y(v) => *v,
            Fov::X(v) => *v,
        }
    }
}

impl Camera {
    pub fn new(img: Vector2<u32>, fov: Fov, mat: Matrix4<f32>, flip: bool) -> Camera {
        let to_world = mat;
        let to_local = to_world.inverse_transform().unwrap();

        // Control the flipping on the horizontal axis
        let x_v = if flip { 1.0 } else { -1.0 };

        // Adjust the fov variables
        let aspect_ratio = img.x as f32 / img.y as f32;
        //2.0 * f32::tan((fov / 2.0) * f32::consts::PI / 180.0));//(fov * f32::consts::PI / 180.0);
        let fov_rad = match fov {
            Fov::X(v) => Rad(v * (1.0 / aspect_ratio) * f32::consts::PI / 180.0),
            Fov::Y(v) => Rad(v * aspect_ratio * f32::consts::PI / 180.0),
        };

        // Compute camera informations
        let camera_to_sample = Matrix4::from_nonuniform_scale(-0.5, -0.5 * aspect_ratio, 1.0)
            * Matrix4::from_translation(Vector3::new(-1.0, -1.0 / aspect_ratio, 0.0))
            * perspective(fov_rad, 1.0, 1e-2, 1000.0)
            * Matrix4::from_nonuniform_scale(x_v, 1.0, -1.0); // undo gluPerspective (z neg)
        let sample_to_camera = camera_to_sample.inverse_transform().unwrap();

        // Compute the image plane inside the sample space.
        let p0 = sample_to_camera.transform_point(Point3::new(0.0, 0.0, 0.0));
        let p1 = sample_to_camera.transform_point(Point3::new(1.0, 1.0, 0.0));
        let image_rect_min = Point2::new(p0.x.min(p1.x), p0.y.min(p1.y)) / p0.z.min(p1.z);
        let image_rect_max = Point2::new(p0.x.max(p1.x), p0.y.max(p1.y)) / p0.z.max(p1.z);
        Camera {
            img,
            camera_to_sample,
            sample_to_camera,
            to_world,
            to_local,
            image_rect_min,
            image_rect_max,
        }
    }

    pub fn size(&self) -> &Vector2<u32> {
        &self.img
    }

    pub fn scale_image(&mut self, s: f32) {
        self.img = Vector2::new(
            (s * self.img.x as f32) as u32,
            (s * self.img.y as f32) as u32,
        );
    }

    /// Compute the ray direction going through the pixel passed
    pub fn generate(&self, px: Point2<f32>) -> Ray {
        let near_p = self.sample_to_camera.transform_point(Point3::new(
            px.x / (self.img.x as f32),
            px.y / (self.img.y as f32),
            0.0,
        ));
        let d = near_p.to_vec().normalize();
        // info!("d: {:?}",  self.to_world.transform_vector(d));

        Ray::new(self.position(), self.to_world.transform_vector(d))
    }

    /// Method to splat a given sample on the camera
    pub fn sample_direct(&self, p: &Point3<f32>) -> Option<(Color, Point2<f32>)> {
        let ref_p = self.to_local.transform_point(*p);
        if ref_p.z < 0.0 {
            return None;
        }

        let screen_pos = self.camera_to_sample.transform_point(ref_p);
        if screen_pos.x < 0.0 || screen_pos.x > 1.0 || screen_pos.y < 0.0 || screen_pos.y > 1.0 {
            return None;
        }
        let screen_pos = Point2::new(
            screen_pos.x * self.img.x as f32,
            screen_pos.y * self.img.y as f32,
        );
        let mut local_d = ref_p.to_vec();
        let inv_dist = 1.0 / local_d.magnitude();
        local_d *= inv_dist;

        let importance = self.importance(local_d);
        if importance == 0.0 {
            None
        } else {
            Some((Color::value(importance) * inv_dist * inv_dist, screen_pos))
        }
    }

    fn importance(&self, d: Vector3<f32>) -> f32 {
        let cos_theta = d.z;
        if cos_theta <= 0.0 {
            return 0.0;
        }
        let inv_cos_theta = 1.0 / cos_theta;
        let p = Point2::new(d.x * inv_cos_theta, d.y * inv_cos_theta);
        if p.x < self.image_rect_min.x
            || p.x > self.image_rect_max.x
            || p.y < self.image_rect_min.y
            || p.x > self.image_rect_max.y
        {
            return 0.0;
        }

        let size = (self.image_rect_max.x - self.image_rect_min.x)
            * (self.image_rect_max.y - self.image_rect_min.y);
        (1.0 / size as f32) * inv_cos_theta * inv_cos_theta * inv_cos_theta
    }

    pub fn position(&self) -> Point3<f32> {
        self.to_world.transform_point(Point3::new(0.0, 0.0, 0.0))
    }

    pub fn print_info(&self) {
        let pix = Point2::new(self.img.x as f32 * 0.5 + 0.5, self.img.y as f32 * 0.5 + 0.5);
        let view_dir = self.generate(pix).d;
        info!(" - Position: {:?}", self.position());
        info!(" - View direction: {:?}", view_dir);
    }
}
