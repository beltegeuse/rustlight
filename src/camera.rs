use cgmath::*;
use std::f32;
// my includes
use structure::Ray;

#[derive(Serialize, Deserialize, Debug)]
pub struct CameraParam {
    pub pos: Point3<f32>,
    pub dir: Vector3<f32>,
    pub up: Vector3<f32>,
    pub img: Vector2<u32>,
    pub fov: f32,
}

pub struct Camera {
    pub param: CameraParam,
    // Internally
    dir_top_left: Vector3<f32>,
    screen_du: Vector3<f32>,
    screen_dv: Vector3<f32>,
}

impl Camera {
    pub fn new(param: CameraParam) -> Camera {
        let dz = param.dir.normalize();
        let dx = -dz.cross(param.up).normalize();
        let dy = dx.cross(dz).normalize();
        let dim_y = 2.0 * f32::tan((param.fov / 2.0) * f32::consts::PI / 180.0);
        let aspect_ratio = param.img.x as f32 / param.img.y as f32;
        let dim_x = dim_y * aspect_ratio;
        let screen_du = dx * dim_x;
        let screen_dv = dy * dim_y;
        let dir_top_left = dz - 0.5 * screen_du - 0.5 * screen_dv;
        Camera {
            param,
            dir_top_left,
            screen_du,
            screen_dv,
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
        let d = (self.dir_top_left + px.x / (self.param.img.x as f32) * self.screen_du
            + px.y / (self.param.img.y as f32) * self.screen_dv)
            .normalize();

        Ray::new(self.param.pos, d)
    }
}
