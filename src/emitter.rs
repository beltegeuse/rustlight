use crate::structure::*;
use cgmath::*;

pub struct LightSampling<'a> {
    pub emitter: &'a Emitter,
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

pub struct LightSamplingPDF<'a> {
    pub emitter: &'a Emitter,
    pub o: Point3<f32>,
    pub p: Point3<f32>,
    pub n: Vector3<f32>,
    pub dir: Vector3<f32>,
}

impl<'a> LightSamplingPDF<'a> {
    pub fn new(ray: &Ray, its: &'a Intersection) -> LightSamplingPDF<'a> {
        LightSamplingPDF {
            emitter: &its.mesh,
            o: ray.o,
            p: its.p,
            n: its.n_g, // FIXME: Geometrical normal?
            dir: ray.d,
        }
    }
}

pub trait Emitter {
	fn sample_position(&self,
        v1: f32,
        v2: f32,
        uv: Point2<f32>) -> (PDF, SampledPosition);
	fn direct_pdf(&self, light_sampling: &LightSamplingPDF) -> PDF;
	fn sample_direct(&self,
        p: &Point3<f32>,
        r_sel: f32,
        r: f32,
        uv: Point2<f32>) -> LightSampling;
	fn flux(&self) -> Color;
	fn emitted_luminance(&self, ray: &Ray) -> Color;
}


pub struct EnvironmentLight {
    pub luminance: Color,
    world_radius: f32,
}
impl Emitter for EnvironmentLight {
	fn sample_position(&self,
        v1: f32,
        v2: f32,
        uv: Point2<f32>) -> (PDF, SampledPosition) {
			unimplemented!();
		}
	fn direct_pdf(&self, light_sampling: &LightSamplingPDF) -> PDF {
		unimplemented!();
	}
	fn sample_direct(&self,
        p: &Point3<f32>,
        r_sel: f32,
        r: f32,
        uv: Point2<f32>) -> LightSampling {
			unimplemented!();
		}
	fn flux(&self) -> Color {
		std::f32::consts::PI * self.world_radius.powi(2) * self.luminance
	}
	fn emitted_luminance(&self, ray: &Ray) -> Color {
		self.luminance
	}
}
