use crate::integrators::*;
use crate::math::*;

pub struct IntegratorAO {
    pub max_distance: Option<f32>,
    pub normal_correction: bool,
}

impl Integrator for IntegratorAO {
    fn compute(&mut self, sampler: &mut dyn Sampler, accel: &dyn Acceleration, scene: &Scene) -> BufferCollection {
        compute_mc(self, sampler, accel, scene)
    }
}
impl IntegratorMC for IntegratorAO {
    fn compute_pixel(
        &self,
        (ix, iy): (u32, u32),
        accel: &dyn Acceleration,
        scene: &Scene,
        sampler: &mut dyn Sampler,
        _: &EmitterSampler,
    ) -> Color {
        let pix = Point2::new(ix as f32 + sampler.next(), iy as f32 + sampler.next());
        let ray = scene.camera.generate(pix);

        // Do the intersection for the first path
        let its = match accel.trace(&ray) {
            Some(its) => its,
            None => return Color::zero(),
        };

        // Compute an new direction
        // Note that we do not flip the normal automatically,
        // for the light definition (only one sided)
        if !self.normal_correction && its.cos_theta() <= 0.0 {
            return Color::zero();
        }
        let flipped = self.normal_correction && its.cos_theta() <= 0.0;
        let d_local = cosine_sample_hemisphere(sampler.next2d());
        let d_world = if flipped {
            its.frame.to_world(-d_local)
        } else {
            its.frame.to_world(d_local)
        };

        // Check the new intersection distance
        let ray = Ray::new(its.p, d_world);
        match accel.trace(&ray) {
            None => Color::one(),
            Some(new_its) => match self.max_distance {
                None => Color::zero(),
                Some(d) => {
                    if new_its.dist > d {
                        Color::one()
                    } else {
                        Color::zero()
                    }
                }
            },
        }
    }
}
