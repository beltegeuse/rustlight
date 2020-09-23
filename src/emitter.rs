use crate::geometry::Mesh;
use crate::math::{sample_uniform_sphere, Distribution1D};
use crate::structure::*;
use cgmath::*;
use std::sync::Arc;

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
            n: its.n_g,
            dir: ray.d,
        }
    }
}

pub trait Emitter: Send + Sync {
    /// Direct sampling & PDF methods
    fn direct_pdf(&self, light_sampling: &LightSamplingPDF) -> PDF;
    fn sample_direct(&self, p: &Point3<f32>, r: f32, uv: Point2<f32>) -> LightSampling;

    /// Sample a particular point on the light source
    fn sample_position(&self, s: f32, uv: Point2<f32>) -> (SampledPosition, Color);

    /// Emitters attributes
    fn flux(&self) -> Color;
    fn emitted_luminance(&self, d: Vector3<f32>) -> Color;
}

pub struct EnvironmentLight {
    pub luminance: Color,
    pub world_radius: f32,
    pub world_position: Point3<f32>,
}
impl Emitter for EnvironmentLight {
    fn sample_position(&self, _s: f32, uv: Point2<f32>) -> (SampledPosition, Color) {
        // TODO: Check this function
        let d = sample_uniform_sphere(uv);
        let pdf = 1.0 / (self.world_radius * self.world_radius * std::f32::consts::PI * 4.0);

        // TODO: Should not be this!
        (
            SampledPosition {
                p: self.world_position + d * self.world_radius,
                n: -d,
                pdf: PDF::Area(pdf),
            },
            self.flux(),
        )
    }
    fn direct_pdf(&self, _light_sampling: &LightSamplingPDF) -> PDF {
        unimplemented!();
    }
    fn sample_direct(&self, _p: &Point3<f32>, _r: f32, _uv: Point2<f32>) -> LightSampling {
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
        self.cdf.as_ref().unwrap().normalization * self.emission * std::f32::consts::PI
    }

    fn emitted_luminance(&self, _d: Vector3<f32>) -> Color {
        self.emission
    }

    fn sample_direct(&self, p: &Point3<f32>, r: f32, uv: Point2<f32>) -> LightSampling {
        let sampled_pos = self.sample(r, uv);

        // Compute the distance
        let mut d: Vector3<f32> = sampled_pos.p - p;
        let dist = d.magnitude();
        if dist != 0.0 {
            d /= dist;
        }

        // Compute Geometry factor
        let geom = if dist != 0.0 {
            sampled_pos.n.dot(-d).max(0.0) / (dist * dist)
        } else {
            0.0
        };

        // PDF & Weight
        let pdf_area = sampled_pos.pdf.value();
        let pdf = sampled_pos.pdf.as_solid_angle_geom(geom);
        let weight = if pdf.is_zero() {
            Color::zero()
        } else {
            self.emission * geom / pdf_area
        };

        LightSampling {
            emitter: self,
            pdf,
            p: sampled_pos.p,
            n: sampled_pos.n,
            d,
            weight,
        }
    }

    fn sample_position(&self, s: f32, uv: Point2<f32>) -> (SampledPosition, Color) {
        let res = self.sample(s, uv);
        let w = self.emission / res.pdf.value();
        (self.sample(s, uv), w)
    }
}

pub struct EmitterSampler {
    pub emitters: Vec<Arc<dyn Emitter>>,
    pub emitters_cdf: Distribution1D,
}

impl EmitterSampler {
    pub fn pdf(&self, emitter: &dyn Emitter) -> f32 {
        let emitter_addr: [usize; 2] = unsafe { std::mem::transmute(emitter) };
        for (i, e) in self.emitters.iter().enumerate() {
            let other_addr: [usize; 2] = unsafe { std::mem::transmute(e.as_ref()) };
            if emitter_addr[0] == other_addr[0] {
                //if std::ptr::eq(emitter, *e) {
                // I need the index to retrive an info
                // This info cannot be stored inside the Emitter
                return self.emitters_cdf.pdf(i);
            }
        }

        // For debug
        println!("Size: {}", self.emitters.len());
        for e in &self.emitters {
            println!(" - {:p} != {:p}", (*e), emitter);
        }

        panic!("Impossible to found the emitter: {:p}", emitter);
    }

    pub fn direct_pdf(&self, emitter: &dyn Emitter, light_sampling: &LightSamplingPDF) -> PDF {
        emitter.direct_pdf(light_sampling) * self.pdf(emitter)
    }

    pub fn sample_light(
        &self,
        p: &Point3<f32>,
        r_sel: f32,
        r: f32,
        uv: Point2<f32>,
    ) -> LightSampling {
        // Select the point on the light
        let (pdf_sel, emitter) = self.random_select_emitter(r_sel);
        let mut res = emitter.sample_direct(p, r, uv);
        res.weight /= pdf_sel;
        res.pdf = res.pdf * pdf_sel;
        res
    }
    pub fn random_select_emitter(&self, v: f32) -> (f32, &dyn Emitter) {
        let id_light = self.emitters_cdf.sample(v);
        (
            self.emitters_cdf.pdf(id_light),
            self.emitters[id_light].as_ref(),
        )
    }

    pub fn random_sample_emitter_position(
        &self,
        v1: f32,
        v2: f32,
        uv: Point2<f32>,
    ) -> (&dyn Emitter, SampledPosition, Color) {
        let (pdf_sel, emitter) = self.random_select_emitter(v1);
        let (mut sampled_pos, w) = emitter.sample_position(v2, uv);
        sampled_pos.pdf = sampled_pos.pdf * pdf_sel;
        (emitter, sampled_pos, w / pdf_sel)
    }
}
