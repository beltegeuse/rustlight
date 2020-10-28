use crate::geometry::Mesh;
use crate::math::{sample_uniform_sphere, Distribution1D, Distribution2D};
use crate::scene::Scene;
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
    fn direct_sample(&self, p: &Point3<f32>, r: f32, uv: Point2<f32>) -> LightSampling;

    /// Sample a particular point on the light source
    fn sample_position(&self, s: f32, uv: Point2<f32>) -> (SampledPosition, Color);
    fn sample_direction(
        &self,
        sampled_pos: &SampledPosition,
        d: Point2<f32>,
    ) -> (Vector3<f32>, PDF, Color);

    /// Emitters attributes
    fn flux(&self) -> Color;
    fn eval(&self, d: Vector3<f32>) -> Color;

    // Building scene dependent information
    fn preprocess(&mut self, _: &Scene) {}
}

pub struct DirectionalLight {
    // Light direction (from the light to the world)
    pub direction: Vector3<f32>,
    // Light intensity
    pub intensity: Color,
    // World sphere (populated with preprocess)
    pub bsphere: Option<BoundingSphere>,
}

impl Emitter for DirectionalLight {
    fn preprocess(&mut self, scene: &Scene) {
        self.bsphere = scene.bsphere.clone();
        self.bsphere.as_mut().unwrap().radius *= 1.1;
    }

    fn direct_pdf(&self, _: &LightSamplingPDF) -> PDF {
        PDF::Discrete(1.0) // Deterministic sampling, no decision
    }

    fn direct_sample(&self, v: &Point3<f32>, _: f32, _: Point2<f32>) -> LightSampling {
        let bsphere = self.bsphere.as_ref().unwrap();

        // TODO: Do the test from Mitsuba (distance?)
        //  To avoid self intersections
        // let disk_center = bsphere.center - self.direction*bsphere.radius;

        let p = v - bsphere.radius * self.direction;
        LightSampling {
            emitter: self,
            pdf: PDF::Discrete(1.0),
            p: Point3::new(p.x, p.y, p.z),
            n: self.direction,
            d: -self.direction,
            weight: self.intensity,
        }
    }

    fn sample_position(&self, _s: f32, uv: Point2<f32>) -> (SampledPosition, Color) {
        let bsphere = self.bsphere.as_ref().unwrap();

        // Sampling a disk
        let p = crate::math::concentric_sample_disk(uv);
        let area = std::f32::consts::PI * bsphere.radius.powi(2);

        // Compute the point offset
        let frame = crate::math::Frame::new(self.direction);
        let poff = frame.to_world(Vector3::new(p.x, p.y, 0.0) * bsphere.radius);

        // Compute position
        // - Offset center
        let p = bsphere.center - self.direction * bsphere.radius;
        // - Move perpendicular
        let p = p + poff;

        (
            SampledPosition {
                p,
                n: self.direction,
                pdf: PDF::Area(1.0 / area),
            },
            self.intensity * area,
        )
    }

    fn flux(&self) -> Color {
        // TODO: Can be precomputed (as for PDF)
        let area = std::f32::consts::PI * self.bsphere.as_ref().unwrap().radius.powi(2);
        area * self.intensity
    }

    fn eval(&self, _: Vector3<f32>) -> Color {
        self.intensity
    }

    fn sample_direction(
        &self,
        sampled_pos: &SampledPosition,
        _: Point2<f32>,
    ) -> (Vector3<f32>, PDF, Color) {
        (sampled_pos.n, PDF::Discrete(1.0), Color::one())
    }
}

pub struct PointEmitter {
    pub intensity: Color,
    pub position: Point3<f32>,
}
impl Emitter for PointEmitter {
    fn direct_pdf(&self, _: &LightSamplingPDF) -> PDF {
        // TODO: Add domain
        PDF::Discrete(1.0)
    }

    fn direct_sample(&self, v: &Point3<f32>, _r: f32, _uv: Point2<f32>) -> LightSampling {
        let p = self.position;
        let d = p - v;
        let dist = d.magnitude();
        let d = d / dist;

        let pdf = PDF::Discrete(1.0);
        LightSampling {
            emitter: self,
            pdf,
            p,
            // TODO: Not ideal but the normal is not defined on point light
            n: Vector3::new(0.0, 0.0, 0.0),
            d,
            weight: self.intensity / dist.powi(2),
        }
    }

    fn sample_position(&self, _s: f32, _uv: Point2<f32>) -> (SampledPosition, Color) {
        (
            SampledPosition {
                p: self.position,
                // TODO: Not ideal but the normal is not defined on point light
                n: Vector3::new(0.0, 0.0, 0.0),
                pdf: PDF::Discrete(1.0),
            },
            self.intensity * 4.0 * std::f32::consts::PI,
        )
    }

    fn sample_direction(&self, _: &SampledPosition, d: Point2<f32>) -> (Vector3<f32>, PDF, Color) {
        (
            crate::math::sample_uniform_sphere(d),
            PDF::SolidAngle(std::f32::consts::FRAC_1_PI * 0.25),
            Color::one(),
        )
    }

    fn flux(&self) -> Color {
        self.intensity * 4.0 * std::f32::consts::PI
    }

    fn eval(&self, _: Vector3<f32>) -> Color {
        self.intensity
    }
}

pub enum EnvironmentLightColor {
    Constant(Color),
    Texture {
        image: Bitmap,
        image_cdf: Distribution2D,
    },
}
impl std::fmt::Debug for EnvironmentLightColor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            EnvironmentLightColor::Constant(c) => {
                f.write_str(&format!("Constant: color = {:?}", c))
            }
            EnvironmentLightColor::Texture { image_cdf, .. } => f.write_str(&format!(
                "Texture: normalization = {}",
                image_cdf.marginal.func_int
            )),
        }
    }
}
pub fn to_spherical_coordinates(d: Vector3<f32>) -> Vector2<f32> {
    Vector2::new(
        {
            let p = d.y.atan2(d.x);
            if p < 0.0 {
                p + 2.0 * std::f32::consts::PI
            } else {
                p
            }
        } * std::f32::consts::FRAC_1_PI
            * 0.5,
        crate::clamp(d.z, -1.0, 1.0).acos() * std::f32::consts::FRAC_1_PI,
    )
}

impl EnvironmentLightColor {
    // Method to build 2D cdf from the bitmap
    pub fn new_texture(image: Bitmap) -> Self {
        // Precompute the sin theta inside the bitmap
        let mut image_pdf = image.clone();
        for y in 0..image.size.y {
            let w = ((y as f32 + 0.5) * std::f32::consts::PI / image.size.y as f32).sin();
            for x in 0..image.size.x {
                *image_pdf.pixel_mut(Point2::new(x, y)) *= w;
            }
        }

        let image_cdf = Distribution2D::from_bitmap(&image_pdf);
        EnvironmentLightColor::Texture { image, image_cdf }
    }

    fn sample_direction(&self, uv: Point2<f32>) -> (Vector3<f32>, Color, f32) {
        match self {
            EnvironmentLightColor::Constant(c) => (
                sample_uniform_sphere(uv),
                *c,
                1.0 / (std::f32::consts::PI * 4.0),
            ),
            EnvironmentLightColor::Texture { image, image_cdf } => {
                // Return [0, size]
                let uv = image_cdf.sample_continuous(uv);
                // TODO: Do bilinear interpolation
                let value = image.pixel(Point2::new(uv.x as u32, uv.y as u32));
                let pdf = image_cdf.pdf(Point2::new(uv.x as usize, uv.y as usize));

                // Compute spherical coordinates for the direction
                let (sin_phi, cos_phi) =
                    ((2.0 * std::f32::consts::PI / image.size.x as f32) * uv.x).sin_cos();
                let (sin_theta, cos_theta) =
                    ((std::f32::consts::PI / image.size.y as f32) * uv.y).sin_cos();

                let d = Vector3::new(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
                if sin_theta == 0.0 {
                    (d, Color::zero(), 0.0)
                } else {
                    (
                        d,
                        value,
                        pdf / (2.0 * std::f32::consts::PI.powi(2) * sin_theta),
                    )
                }
            }
        }
    }

    fn eval(&self, d: Vector3<f32>) -> Color {
        match self {
            EnvironmentLightColor::Constant(c) => *c,
            EnvironmentLightColor::Texture { image, .. } => {
                let uv = to_spherical_coordinates(d);
                // TODO: Bilinear interpolation
                image.pixel_uv(uv)
            }
        }
    }

    fn pdf(&self, d: Vector3<f32>) -> f32 {
        match self {
            EnvironmentLightColor::Constant(_) => 1.0 / (std::f32::consts::PI * 4.0),
            EnvironmentLightColor::Texture { image_cdf, image } => {
                // Map the direction to [0,1]x[0,1]
                let uv = to_spherical_coordinates(d);
                // Use [0, size] coordinates
                let pdf = image_cdf.pdf(Point2::new(
                    (uv.x * image.size.x as f32) as usize,
                    (uv.y * image.size.y as f32) as usize,
                ));
                // TODO: We do not need to scale / size.y as uv is in [0,1]
                let sin_theta = (std::f32::consts::PI * uv.y).sin();
                if sin_theta == 0.0 {
                    0.0
                } else {
                    pdf / (2.0 * std::f32::consts::PI.powi(2) * sin_theta)
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct EnvironmentLight {
    pub luminance: EnvironmentLightColor,
    pub bsphere: Option<BoundingSphere>,
}
impl EnvironmentLight {}

//  Or something else?
impl Emitter for EnvironmentLight {
    fn preprocess(&mut self, scene: &Scene) {
        self.bsphere = scene.bsphere.clone();
        self.bsphere.as_mut().unwrap().radius *= 1.1;
    }

    // TODO: Not optimal sampling (where the position and direction are decorrelated)
    //  Making a lot of ray missing the target
    fn sample_position(&self, _s: f32, uv: Point2<f32>) -> (SampledPosition, Color) {
        let bsphere = self.bsphere.as_ref().unwrap();

        // Sampling direction
        let d = sample_uniform_sphere(uv);
        let area_sphere = 4.0 * std::f32::consts::PI * bsphere.radius.powi(2);

        // Compute position
        let p = bsphere.center - d * bsphere.radius;

        let inv_pdf = area_sphere;
        (
            SampledPosition {
                p,
                n: d,
                pdf: PDF::Area(1.0 / inv_pdf),
            },
            match &self.luminance {
                EnvironmentLightColor::Constant(c) => *c * inv_pdf * std::f32::consts::PI,
                EnvironmentLightColor::Texture { image_cdf, .. } => {
                    Color::value(inv_pdf / image_cdf.marginal.func_int)
                }
            },
        )
    }
    fn direct_pdf(&self, light_sampling: &LightSamplingPDF) -> PDF {
        // dir is toward the light
        PDF::SolidAngle(self.luminance.pdf(light_sampling.dir))
    }
    fn direct_sample(&self, v: &Point3<f32>, _r: f32, uv: Point2<f32>) -> LightSampling {
        let bsphere = self.bsphere.as_ref().unwrap();

        // TODO: Pass the Option<normal> to have better IS (cosine weighted)
        // The envmap is constant so we need to generate a direction over the sphere
        let (d, color, pdf) = self.luminance.sample_direction(uv);

        let t = bsphere.intersect(&Ray::new(*v, d));
        if t.is_none() {
            warn!("Miss bSphere");
            // TODO: Not ideal to create a dummy structure like this
            return LightSampling {
                emitter: self,
                pdf: PDF::SolidAngle(pdf),
                p: Point3::new(0.0, 0.0, 0.0),
                n: Vector3::new(0.0, 0.0, 0.0),
                d,
                weight: Color::zero(),
            };
        }
        let t = t.unwrap();

        let p = v + d * t;
        let n = (bsphere.center - p).normalize();
        LightSampling {
            emitter: self,
            pdf: PDF::SolidAngle(pdf),
            p: Point3::new(p.x, p.y, p.z),
            n,
            // d is toward the light source
            d,
            weight: color / pdf,
        }
    }
    fn flux(&self) -> Color {
        match &self.luminance {
            EnvironmentLightColor::Constant(c) => {
                std::f32::consts::PI * self.bsphere.as_ref().unwrap().radius.powi(2) * c
            }
            EnvironmentLightColor::Texture { image_cdf, .. } => Color::value(
                std::f32::consts::PI
                    * self.bsphere.as_ref().unwrap().radius.powi(2)
                    // Avg luminance
                    * image_cdf.marginal.func_int,
            ),
        }
    }
    fn eval(&self, d: Vector3<f32>) -> Color {
        self.luminance.eval(d)
    }

    fn sample_direction(
        &self,
        sampled_pos: &SampledPosition,
        d: Point2<f32>,
    ) -> (Vector3<f32>, PDF, Color) {
        match &self.luminance {
            EnvironmentLightColor::Constant(_) => {
                // Sample the direction with cos IS
                // This strategy does not increase the variance
                let d_out = crate::math::cosine_sample_hemisphere(d);
                let (weight, pdf) = if d_out.z < 0.0 {
                    // Can be due to f32 inaccuracies
                    (Color::zero(), PDF::SolidAngle(0.0))
                } else {
                    (
                        Color::one(),
                        PDF::SolidAngle(d_out.z * std::f32::consts::FRAC_1_PI),
                    )
                };

                let frame = crate::math::Frame::new(sampled_pos.n);
                let d_out_global = frame.to_world(d_out);
                (d_out_global, pdf, weight)
            }
            EnvironmentLightColor::Texture { image_cdf, .. } => {
                let (d, color, pdf) = self.luminance.sample_direction(d);
                (
                    d,
                    PDF::SolidAngle(pdf),
                    (color * image_cdf.marginal.func_int) / pdf,
                )
            }
        }
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
        self.cdf.as_ref().unwrap().total() * self.emission * std::f32::consts::PI
    }

    fn eval(&self, _d: Vector3<f32>) -> Color {
        self.emission
    }

    fn direct_sample(&self, p: &Point3<f32>, r: f32, uv: Point2<f32>) -> LightSampling {
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
        (self.sample(s, uv), self.flux())
    }

    fn sample_direction(
        &self,
        sampled_pos: &SampledPosition,
        d: Point2<f32>,
    ) -> (Vector3<f32>, PDF, Color) {
        let d_out = crate::math::cosine_sample_hemisphere(d);
        let (weight, pdf) = if d_out.z < 0.0 {
            // Can be due to f32 inaccuracies
            (Color::zero(), PDF::SolidAngle(0.0))
        } else {
            // The weight is one as the cosine factor is perfectly IS
            (
                Color::one(),
                PDF::SolidAngle(d_out.z * std::f32::consts::FRAC_1_PI),
            )
        };

        let frame = crate::math::Frame::new(sampled_pos.n);
        let d_out_global = frame.to_world(d_out);
        (d_out_global, pdf, weight)
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
        let mut res = emitter.direct_sample(p, r, uv);
        res.weight /= pdf_sel;
        res.pdf = res.pdf * pdf_sel;
        res
    }
    pub fn random_select_emitter(&self, v: f32) -> (f32, &dyn Emitter) {
        let id_light = self.emitters_cdf.sample_discrete(v);
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
