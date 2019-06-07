use crate::geometry::Mesh;

use crate::math::sample_uniform_sphere;
use crate::structure::*;
use cgmath::*;
pub struct LightSampling<'a> {
    pub emitter: &'a dyn Emitter,
    pub pdf: PDF,
    pub p: Point3<f32>,
    pub n: Vector3<f32>,
    pub d: Vector3<f32>,
    pub weight: Color,
}
impl<'a> LightSampling<'a> {
    pub fn is_valid(&'a self) -> bool {
        !self.pdf.is_zero()
    }
}

pub struct LightSamplingPDF {
    pub o: Point3<f32>,
    pub p: Point3<f32>,
    pub n: Vector3<f32>,
    pub dir: Vector3<f32>,
}

impl LightSamplingPDF {
    pub fn new(ray: &Ray, its: &Intersection) -> LightSamplingPDF {
        LightSamplingPDF {
            o: ray.o,
            p: its.p,
            n: its.n_g, // FIXME: Geometrical normal?
            dir: ray.d,
        }
    }
}

pub trait Emitter: Send + Sync {
    fn sample_position(&self, s: f32, uv: Point2<f32>) -> SampledPosition;
    fn direct_pdf(&self, light_sampling: &LightSamplingPDF) -> PDF;
    fn sample_direct(&self, p: &Point3<f32>, r: f32, uv: Point2<f32>) -> LightSampling;
    fn flux(&self) -> Color;
    fn emitted_luminance(&self, d: Vector3<f32>) -> Color;
}

pub struct EnvironmentLight {
    pub luminance: Color,
    pub world_radius: f32,
    pub world_position: Point3<f32>,
}
impl Emitter for EnvironmentLight {
    fn sample_position(&self, s: f32, uv: Point2<f32>) -> SampledPosition {
        // TODO: Check this function
        let d = sample_uniform_sphere(uv);
        let pdf = 1.0 / (self.world_radius * self.world_radius * std::f32::consts::PI * 4.0);
        SampledPosition {
            p: self.world_position + d * self.world_radius,
            n: -d,
            pdf: PDF::Area(pdf),
        }
    }
    fn direct_pdf(&self, light_sampling: &LightSamplingPDF) -> PDF {
        unimplemented!();
    }
    fn sample_direct(&self, p: &Point3<f32>, _r: f32, uv: Point2<f32>) -> LightSampling {
        unimplemented!();
    }
    fn flux(&self) -> Color {
        std::f32::consts::PI * self.world_radius.powi(2) * self.luminance
    }
    fn emitted_luminance(&self, _d: Vector3<f32>) -> Color {
        self.luminance
    }
}

impl Emitter for Mesh {
    fn direct_pdf(&self, light_sampling: &LightSamplingPDF) -> PDF {
        let cos_light = light_sampling.n.dot(-light_sampling.dir).max(0.0);
        if cos_light == 0.0 {
            PDF::SolidAngle(0.0)
        } else {
            let geom_inv = (light_sampling.p - light_sampling.o).magnitude2() / cos_light;
            PDF::SolidAngle(self.pdf() * geom_inv) // TODO: Check
        }
    }

    fn flux(&self) -> Color {
        self.cdf.normalization * self.emission * std::f32::consts::PI
    }

    fn emitted_luminance(&self, _d: Vector3<f32>) -> Color {
        self.emission
    }

    fn sample_direct(&self, p: &Point3<f32>, r: f32, uv: Point2<f32>) -> LightSampling {
        let sampled_pos = self.sample(r, uv);

        // Compute the distance
        let mut d: Vector3<f32> = sampled_pos.p - p;
        let dist = d.magnitude();
        d /= dist;

        // Compute the geometry
        let pdf = match sampled_pos.pdf {
            PDF::Area(v) => {
                let cos_light = sampled_pos.n.dot(-d).max(0.0);
                if cos_light == 0.0 {
                    PDF::SolidAngle(0.0)
                } else {
                    // FIXME: Make the conversion as a method
                    PDF::SolidAngle((sampled_pos.pdf.value() * dist * dist) / cos_light)
                }
            }
            PDF::SolidAngle(v) => PDF::SolidAngle(v),
            PDF::Discrete(v) => panic!("Discrete pdf is not handled yet"),
        };

        let emission = if pdf.is_zero() {
            Color::zero()
        } else {
            self.emission / pdf.value()
        };
        LightSampling {
            emitter: self,
            pdf,
            p: sampled_pos.p,
            n: sampled_pos.n,
            d,
            weight: emission,
        }
    }

    fn sample_position(&self, s: f32, uv: Point2<f32>) -> SampledPosition {
        self.sample(s, uv)
    }
}
