use cgmath::*;
use integrators::*;
use math::*;
use structure::*;

pub struct IntegratorAO {
    pub max_distance: Option<f32>,
}

impl Integrator for IntegratorAO {
    fn compute(&mut self, scene: &Scene) -> Bitmap {
        compute_mc(self, scene)
    }
}
impl IntegratorMC for IntegratorAO {
    fn compute_pixel(&self, (ix, iy): (u32, u32), scene: &Scene, sampler: &mut Sampler) -> Color {
        let pix = Point2::new(ix as f32 + sampler.next(), iy as f32 + sampler.next());
        let ray = scene.camera.generate(pix);

        // Do the intersection for the first path
        let its = match scene.trace(&ray) {
            Some(its) => its,
            None => return Color::zero(),
        };

        // Compute an new direction
        // Note that we do not flip the normal automatically,
        // for the light definition (only one sided)
        if its.cos_theta() <= 0.0 {
            return Color::zero();
        }
        let d_local = cosine_sample_hemisphere(sampler.next2d());
        let d_world = its.frame.to_world(d_local);

        // Check the new intersection distance
        let ray = Ray::new(its.p, d_world);
        match scene.trace(&ray) {
            None => Color::one(),
            Some(new_its) => match self.max_distance {
                None => Color::zero(),
                Some(d) => if new_its.dist > d {
                    Color::one()
                } else {
                    Color::zero()
                },
            },
        }
    }
}
