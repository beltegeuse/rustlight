use cgmath::*;
use std::f32;
use structure::{Color, Ray};

#[derive(Serialize, Deserialize, Debug)]
pub struct CameraParam {
    pub pos: Point3<f32>,
    pub dir: Vector3<f32>,
    pub up: Vector3<f32>,
    pub img: Vector2<u32>,
    pub fov: f32, //< y
}

pub struct Camera {
    pub param: CameraParam,
    // Internally
    camera_to_sample: Matrix4<f32>,
    sample_to_camera: Matrix4<f32>,
    to_world: Matrix4<f32>,
    to_local: Matrix4<f32>,
}

impl Camera {
    pub fn new(param: CameraParam) -> Camera {
        let dz = param.dir.normalize();
        let dx = -dz.cross(param.up).normalize();
        let dy = dx.cross(dz).normalize();
        let to_world = Matrix4::from(Matrix3::from_cols(-dx, -dy, -dz));
        let to_local = to_world.inverse_transform().unwrap();

        // Compute camera informations
        // fov: y
        let aspect_ratio = param.img.x as f32 / param.img.y as f32;
        let fov_rad = Rad(param.fov * f32::consts::PI / 180.0);
        let camera_to_sample = Matrix4::from_nonuniform_scale(-0.5, -0.5 * aspect_ratio, 1.0)
            * Matrix4::from_translation(Vector3::new(-1.0, -1.0 / aspect_ratio, 0.0))
            * perspective(fov_rad, aspect_ratio, 0.1, 100.0);
        let sample_to_camera = camera_to_sample.inverse_transform().unwrap();
        Camera {
            param,
            camera_to_sample,
            sample_to_camera,
            to_world,
            to_local,
        }
    }

    pub fn look_at(
        pos: Point3<f32>,
        at: Point3<f32>,
        up: Vector3<f32>,
        img: Vector2<u32>,
        fov: f32,
    ) -> Camera {
        let dir = at - pos;
        let param = CameraParam {
            pos,
            dir,
            up,
            img,
            fov,
        };

        Camera::new(param)
    }

    pub fn size(&self) -> &Vector2<u32> {
        &self.param.img
    }

    pub fn scale_image(&mut self, s: f32) {
        self.param.img = Vector2::new(
            (s * self.param.img.x as f32) as u32,
            (s * self.param.img.y as f32) as u32,
        );
    }

    /// Compute the ray direction going through the pixel passed
    pub fn generate(&self, px: Point2<f32>) -> Ray {
        let near_p = self.sample_to_camera.transform_point(Point3::new(
            px.x / (self.param.img.x as f32),
            px.y / (self.param.img.y as f32),
            0.0,
        ));
        let d = near_p.to_vec().normalize();

        Ray::new(self.param.pos, self.to_world.transform_vector(d))
    }

    /// Method to splat a given sample on the camera
    pub fn sample_direct(&self, p: &Point3<f32>) -> Option<(Color, Point2<f32>)> {
        let ref_p = self.to_local.transform_point(*p);
        if ref_p.z < 0.0 {
            return None;
        }

        let screen_pos = self.camera_to_sample.transform_point(ref_p.clone());
        if screen_pos.x < 0.0 || screen_pos.x > 1.0 || screen_pos.y < 0.0 || screen_pos.y > 1.0 {
            return None;
        }
        let screen_pos = Point2::new(screen_pos.x * self.param.img.x as f32, screen_pos.y * self.param.img.y as f32);

        let mut local_d = ref_p.to_vec();
        let inv_dist = 1.0 / local_d.magnitude();
        local_d *= inv_dist;

        let importance = self.importance(local_d);
        if importance == 0.0 {
            None
        } else {
            Some((Color::value(importance), screen_pos))
        }
    }

    fn importance(&self, d: Vector3<f32>) -> f32 {
        let cos_theta = d.z;
        if cos_theta <= 0.0 {
            return 0.0;
        }
        let inv_cos_theta = 1.0 / cos_theta;
        let p = Point2::new(d.x * inv_cos_theta, d.y * inv_cos_theta);
        if p.x < 0.0 || p.x > self.param.img.x as f32 || p.y < 0.0 || p.x > self.param.img.y as f32 {
            return 0.0;
        }

        return (1.0 / (self.param.img.x * self.param.img.y) as f32) * inv_cos_theta * inv_cos_theta;
    }
}
