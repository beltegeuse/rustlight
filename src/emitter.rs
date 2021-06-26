use crate::constants::ONE_MINUS_EPSILON;
use crate::geometry::{EmissionType, Mesh};
use crate::math::{sample_uniform_sphere, Distribution1D, Distribution2D};
use crate::samplers::Sampler;
use crate::scene::Scene;
use crate::structure::*;
use cgmath::*;
use std::{collections::HashMap, sync::Arc};

pub struct LightSampling<'a> {
    pub emitter: &'a dyn Emitter,
    pub pdf: PDF,
    pub p: Point3<f32>,
    pub n: Vector3<f32>,
    pub uv: Option<Vector2<f32>>,
    pub primitive_id: Option<usize>,
    pub d: Vector3<f32>,
    pub weight: Color,
}
impl<'a> LightSampling<'a> {
    pub fn is_valid(&'a self) -> bool {
        !self.pdf.is_zero()
    }
}

pub struct LightSamplingPDF {
    pub o: Point3<f32>,  // shading point position
    pub p: Point3<f32>,  // position on the light
    pub n: Vector3<f32>, // normal on the light (geometric)
    pub uv: Option<Vector2<f32>>,
    pub dir: Vector3<f32>, // direction
}

impl LightSamplingPDF {
    pub fn new(ray: &Ray, its: &Intersection) -> LightSamplingPDF {
        LightSamplingPDF {
            o: ray.o,
            p: its.p,
            n: its.n_g,
            uv: its.uv,
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
    fn eval(&self, d: Vector3<f32>, uv: Option<Vector2<f32>>) -> Color;

    // Building scene dependent information
    fn preprocess(&mut self, _: &Scene) {}

    // TODO: Dirty fix
    fn correct_flux(&self) -> f32;

    // For ATS
    fn is_surface(&self) -> bool {
        false
    }
    fn convert_light_proxy(&self, _emitter_id: usize) -> Vec<LightProxy> {
        vec![]
    }
    fn direct_sample_tri(
        &self,
        _p: &Point3<f32>,
        _primitive_id: usize,
        _uv: Point2<f32>,
    ) -> LightSampling {
        unimplemented!()
    }
    fn direct_pdf_tri(&self, _light_sampling: &LightSamplingPDF, _id_primitive: usize) -> PDF {
        unimplemented!()
    }
    fn sample_position_tri(
        &self,
        _primitive_id: usize,
        _uv: Point2<f32>,
    ) -> (SampledPosition, Color) {
        unimplemented!()
    }
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
            uv: None,
            primitive_id: None,
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
                uv: None,
                pdf: PDF::Area(1.0 / area),
                primitive_id: None,
            },
            self.intensity * area,
        )
    }

    fn flux(&self) -> Color {
        // TODO: Can be precomputed (as for PDF)
        let area = std::f32::consts::PI * self.bsphere.as_ref().unwrap().radius.powi(2);
        area * self.intensity
    }

    fn correct_flux(&self) -> f32 {
        todo!()
    }

    fn eval(&self, _: Vector3<f32>, _: Option<Vector2<f32>>) -> Color {
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
            uv: None,
            primitive_id: None,
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
                uv: None,
                pdf: PDF::Discrete(1.0),
                primitive_id: None,
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

    fn correct_flux(&self) -> f32 {
        1.0 / (4.0 * std::f32::consts::PI)
    }

    fn eval(&self, _: Vector3<f32>, _: Option<Vector2<f32>>) -> Color {
        self.intensity
    }
}

pub struct PointNormalEmitter {
    pub intensity: Color,
    pub position: Point3<f32>,
    pub normal: Vector3<f32>,
}
impl Emitter for PointNormalEmitter {
    fn direct_pdf(&self, _: &LightSamplingPDF) -> PDF {
        todo!() // Discrete in position...
    }

    fn direct_sample(&self, _v: &Point3<f32>, _r: f32, _uv: Point2<f32>) -> LightSampling {
        todo!();
    }

    fn sample_position(&self, _s: f32, _uv: Point2<f32>) -> (SampledPosition, Color) {
        (
            SampledPosition {
                p: self.position,
                n: self.normal,
                uv: None,
                pdf: PDF::Discrete(1.0), // TODO: Fix that
                primitive_id: None,
            },
            self.flux(),
        )
    }

    fn sample_direction(&self, _: &SampledPosition, _d: Point2<f32>) -> (Vector3<f32>, PDF, Color) {
        todo!()
    }

    fn flux(&self) -> Color {
        self.intensity * 2.0
    }

    fn correct_flux(&self) -> f32 {
        1.0 / 2.0
    }

    fn eval(&self, _: Vector3<f32>, _: Option<Vector2<f32>>) -> Color {
        self.intensity
    }

    fn is_surface(&self) -> bool {
        true // Because of the cosine.
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
    let mut uv = Vector2::new(
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
    );
    // It is possible due to f32, that one of the coordinate of UV's vector is 1
    // breaking the assum
    uv.x = crate::clamp(uv.x, 0.0, ONE_MINUS_EPSILON);
    uv.y = crate::clamp(uv.y, 0.0, ONE_MINUS_EPSILON);
    uv
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
                let mut uv = image_cdf.sample_continuous(uv);
                uv.x = crate::clamp(uv.x, 0.0, image.size.x as f32 - 1.0);
                uv.y = crate::clamp(uv.y, 0.0, image.size.y as f32 - 1.0);

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
                uv: None,
                pdf: PDF::Area(1.0 / inv_pdf),
                primitive_id: None,
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
                primitive_id: None,
                uv: None,
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
            uv: None,
            primitive_id: None,
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

    fn correct_flux(&self) -> f32 {
        todo!()
    }

    fn eval(&self, d: Vector3<f32>, _: Option<Vector2<f32>>) -> Color {
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
            let geom = cos_light / (light_sampling.p - light_sampling.o).magnitude2();
            PDF::SolidAngle(self.pdf() / geom)
        }
    }

    fn direct_pdf_tri(&self, light_sampling: &LightSamplingPDF, id_primitive: usize) -> PDF {
        let cos_light = light_sampling.n.dot(-light_sampling.dir).max(0.0);
        if cos_light == 0.0 {
            PDF::SolidAngle(0.0)
        } else {
            let geom = cos_light / (light_sampling.p - light_sampling.o).magnitude2();
            PDF::SolidAngle(self.pdf_tri(id_primitive) / geom)
        }
    }

    fn flux(&self) -> Color {
        let e = match &self.emission {
            EmissionType::Zero => Color::zero(),
            EmissionType::Color { v } => *v,
            EmissionType::HSV { scale } => Color::value(*scale), // TODO
            EmissionType::Texture { scale, .. } => Color::value(*scale), // TODO
        };
        self.cdf.as_ref().unwrap().total() * e * std::f32::consts::PI
    }

    fn correct_flux(&self) -> f32 {
        1.0 / (std::f32::consts::PI)
    }

    fn eval(&self, _d: Vector3<f32>, uv: Option<Vector2<f32>>) -> Color {
        self.emit(&uv)
    }

    fn direct_sample_tri(
        &self,
        p: &Point3<f32>,
        primitive_id: usize,
        uv: Point2<f32>,
    ) -> LightSampling {
        let sampled_pos = self.sample_tri(primitive_id, uv);

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
            self.emit(&sampled_pos.uv) * geom / pdf_area
        };

        LightSampling {
            emitter: self,
            pdf,
            p: sampled_pos.p,
            n: sampled_pos.n,
            uv: sampled_pos.uv,
            primitive_id: None, // Not sampled a particular primitive
            d,
            weight,
        }
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
            self.emit(&sampled_pos.uv) * geom / pdf_area
        };

        LightSampling {
            emitter: self,
            pdf,
            p: sampled_pos.p,
            n: sampled_pos.n,
            uv: sampled_pos.uv,
            primitive_id: sampled_pos.primitive_id,
            d,
            weight,
        }
    }

    fn sample_position_tri(
        &self,
        primitive_id: usize,
        uv: Point2<f32>,
    ) -> (SampledPosition, Color) {
        let sampled_pos = self.sample_tri(primitive_id, uv);
        let phi = self.emit(&sampled_pos.uv) * std::f32::consts::PI / sampled_pos.pdf.value();
        (sampled_pos, phi)
    }
    fn sample_position(&self, s: f32, uv: Point2<f32>) -> (SampledPosition, Color) {
        let sampled_pos = self.sample(s, uv);
        let phi = self.emit(&sampled_pos.uv) * std::f32::consts::PI / sampled_pos.pdf.value();
        (sampled_pos, phi)
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

    fn is_surface(&self) -> bool {
        true
    }
    fn convert_light_proxy(&self, emitter_id: usize) -> Vec<LightProxy> {
        self.indices
            .iter()
            .enumerate()
            .map(|(i, idx)| {
                // Get vertices and compute the normal
                let v0 = self.vertices[idx.x];
                let v1 = self.vertices[idx.y];
                let v2 = self.vertices[idx.z];
                let n = (v1 - v0).cross(v2 - v0);

                // TODO: For now we interpolate at the middle
                let uv = match &self.uv {
                    None => None,
                    Some(uv) => {
                        let uv0 = uv[idx.x];
                        let uv1 = uv[idx.y];
                        let uv2 = uv[idx.z];
                        Some((uv0 + uv1 + uv2) / 3.0)
                    }
                };

                // Construct bound
                let w = n.normalize();
                let theta_o = 0.0;
                let theta_e = std::f32::consts::FRAC_PI_2;
                let phi = self.emit(&uv).channel_max() * n.magnitude() * 0.5;
                let aabb = AABB::default().union_vec(&v0).union_vec(&v1).union_vec(&v2);
                let bounds = LightBounds {
                    aabb: aabb.clone(),
                    w,
                    phi,
                    theta_o,
                    theta_e,
                    cos_theta_o: theta_o.cos(),
                    cos_theta_e: theta_e.cos(),
                    two_sided: false,
                    number_lights: 1,
                    phi_sqr: phi.powi(2),
                    bsphere: aabb.to_sphere(),
                };

                LightProxy {
                    emitter_id,
                    primitive_idx: i,
                    bounds,
                }
            })
            .collect()
    }
}

#[derive(Clone)]
pub struct DirectionCone {
    pub w: Vector3<f32>,
    pub cos_theta: f32,
    pub empty: bool,
}

fn safe_acos(v: f32) -> f32 {
    v.max(-1.0).min(1.0).acos()
}
fn safe_asin(v: f32) -> f32 {
    v.max(-1.0).min(1.0).asin()
}

fn angle_between(v1: &Vector3<f32>, v2: &Vector3<f32>) -> f32 {
    if v1.dot(*v2) < 0.0 {
        std::f32::consts::PI - 2.0 * safe_asin((v2 + v1).magnitude() / 2.0)
    } else {
        2.0 * safe_asin((v2 - v1).magnitude() / 2.0)
    }
}

fn rotate(sin_theta: f32, cos_theta: f32, axis: &Vector3<f32>) -> Matrix4<f32> {
    let a = axis.normalize();

    let c0r0 = a.x * a.x + (1.0 - a.x * a.x) * cos_theta;
    let c0r1 = a.x * a.y * (1.0 - cos_theta) - a.z * sin_theta;
    let c0r2 = a.x * a.z * (1.0 - cos_theta) + a.y * sin_theta;

    // Compute rotations of second and third basis vectors
    let c1r0 = a.x * a.y * (1.0 - cos_theta) + a.z * sin_theta;
    let c1r1 = a.y * a.y + (1.0 - a.y * a.y) * cos_theta;
    let c1r2 = a.y * a.z * (1.0 - cos_theta) - a.x * sin_theta;

    let c2r0 = a.x * a.z * (1.0 - cos_theta) - a.y * sin_theta;
    let c2r1 = a.y * a.z * (1.0 - cos_theta) + a.x * sin_theta;
    let c2r2 = a.z * a.z + (1.0 - a.z * a.z) * cos_theta;

    Matrix4::new(
        c0r0, c0r1, c0r2, 0.0, c1r0, c1r1, c1r2, 0.0, c2r0, c2r1, c2r2, 0.0, 0.0, 0.0, 0.0, 1.0,
    )
    .transpose()
}
fn rotate_angle_axis(theta: f32, axis: &Vector3<f32>) -> Matrix4<f32> {
    let sin_theta = theta.to_radians().sin();
    let cos_theta = theta.to_radians().cos();
    rotate(sin_theta, cos_theta, axis)
}

impl DirectionCone {
    pub fn entire_sphere() -> Self {
        Self {
            w: Vector3::new(0.0, 0.0, 1.0),
            cos_theta: -1.0,
            empty: false,
        }
    }

    pub fn subtended_directions(b: &AABB, p: &Point3<f32>) -> Self {
        let b_sphere = b.to_sphere();
        if (p - b_sphere.center).magnitude2() < b_sphere.radius * b_sphere.radius {
            Self::entire_sphere()
        } else {
            let w = (b_sphere.center - p).normalize();
            let sin_theta_max_2 =
                b_sphere.radius * b_sphere.radius / (b_sphere.center - p).magnitude2();
            let cos_theta = (1.0 - sin_theta_max_2).max(0.0).sqrt();
            Self {
                w,
                cos_theta,
                empty: false,
            }
        }
    }

    pub fn union(a: &Self, b: &Self) -> Self {
        // Handle the cases where one or both cones are empty
        if a.empty {
            return b.clone();
        }
        if b.empty {
            return a.clone();
        }

        // Handle the cases where one cone is inside the other
        let theta_a = safe_acos(a.cos_theta);
        let theta_b = safe_acos(b.cos_theta);
        let theta_d = angle_between(&a.w, &b.w);
        if (theta_d + theta_b).min(std::f32::consts::PI) <= theta_a {
            return a.clone();
        }
        if (theta_d + theta_a).min(std::f32::consts::PI) <= theta_b {
            return b.clone();
        }

        // Compute the spread angle of the merged cone, $\theta_o$
        let theta_o = (theta_a + theta_d + theta_b) / 2.0;
        if theta_o >= std::f32::consts::PI {
            // To big: The entire sphere
            return DirectionCone::entire_sphere();
        }

        // Find the merged cone's axis and return cone union
        let theta_r = theta_o - theta_a;
        let wr = a.w.cross(b.w);
        if wr.magnitude2() == 0.0 {
            DirectionCone::entire_sphere()
        } else {
            let m = rotate_angle_axis(theta_r.to_degrees(), &wr);
            let w = m.transform_vector(a.w);
            DirectionCone {
                w,
                cos_theta: theta_o.cos(),
                empty: false,
            }
        }
    }
}

#[derive(Clone)]
pub struct LightBounds {
    pub aabb: AABB,
    pub w: Vector3<f32>,
    pub phi: f32,
    pub theta_o: f32,
    pub theta_e: f32,
    pub cos_theta_o: f32,
    pub cos_theta_e: f32,
    pub two_sided: bool,
    // For variance computations
    pub number_lights: usize,
    pub phi_sqr: f32,
    pub bsphere: BoundingSphere,
}

impl Default for LightBounds {
    fn default() -> Self {
        Self {
            aabb: AABB::default(),
            w: Vector3::new(0.0, 0.0, 1.0),
            phi: 0.0,
            theta_o: 0.0,
            theta_e: 0.0,
            cos_theta_o: 1.0,
            cos_theta_e: 1.0,
            two_sided: false,
            number_lights: 0,
            phi_sqr: 0.0,
            bsphere: BoundingSphere {
                center: Point3::new(0.0, 0.0, 0.0),
                radius: 0.0,
            },
        }
    }
}

const EPSILON_ATS: f32 = 0.0001;
impl LightBounds {
    pub fn to_dircone(&self) -> DirectionCone {
        DirectionCone {
            w: self.w,
            cos_theta: self.cos_theta_o,
            empty: false,
        }
    }

    pub fn union(a: &LightBounds, b: &LightBounds) -> LightBounds {
        if a.phi == 0.0 {
            return b.clone();
        }
        if b.phi == 0.0 {
            return a.clone();
        }

        let c = DirectionCone::union(&a.to_dircone(), &b.to_dircone());
        let theta_o = safe_acos(c.cos_theta);
        let theta_e = a.theta_e.max(b.theta_e);
        let aabb = a.aabb.union_aabb(&b.aabb);
        LightBounds {
            aabb: aabb.clone(),
            w: c.w,
            phi: a.phi + b.phi,
            theta_o,
            theta_e,
            cos_theta_o: theta_o.cos(),
            cos_theta_e: theta_e.cos(),
            two_sided: a.two_sided | b.two_sided,
            number_lights: a.number_lights + b.number_lights,
            phi_sqr: a.phi_sqr + b.phi_sqr,
            bsphere: aabb.to_sphere(),
        }
    }

    pub fn importance_ray(&self, ray: &Ray, max_dist: Option<f32>) -> f32 {
        /////// Compute distance min: d_min
        let pc = Point3::from_vec(self.aabb.center());
        let (closest, d2) = crate::math::closest_squared_distance_ray_point(ray, max_dist, &pc);
        let d = d2.max(EPSILON_ATS).sqrt();

        ////// Compute the min angle: theta_min
        //  minimum angle between ray and cluster's axis (w)
        let v0 = (ray.o - pc).normalize();
        let v1 = match max_dist {
            Some(max_dist) => ((ray.o + ray.d * max_dist) - pc).normalize(),
            None => -ray.d, // Infinitely away
        };

        // 1) Compute an orthogonal frame where two of the axis (o0 and o1)
        //  contains (v0, v1)
        //  a) Compute o0
        let o0 = v0;
        // Compute up vector
        let up = v0.cross(v1).normalize();
        // Then redo the cross to get o1
        let o1 = up.cross(v0); // Make dot(o0, o1) = 0

        // Check
        if o1.dot(v1) < 0.0 {
            warn!("Wrong sign: o1.v1 < 0: {}", o1.dot(v1));
        }

        // Compute cos_phi_0 (eq. 5) where a is self.w (axis)
        let dot_o0 = o0.dot(self.w);
        let dot_o1 = o1.dot(self.w);
        let length_1 = (dot_o0 * dot_o0 + dot_o1 * dot_o1).max(0.0).sqrt();
        let cos_phi_0 = dot_o0 / length_1;
        assert!(cos_phi_0.is_finite());

        // Compute cos_theta_min (depending if cos_phi_0 is one the segment formed by [v0, v1])
        let cos_theta_min = if dot_o1 < 0.0 || v0.dot(v1) < cos_phi_0 {
            v0.dot(self.w).max(v1.dot(self.w))
        } else {
            let sin_phi_0 = (1.0 - cos_phi_0 * cos_phi_0).sqrt();
            (o0 * cos_phi_0 + o1 * sin_phi_0).dot(self.w)
        };
        let theta_min = crate::math::acos_fast(cos_theta_min);

        // However, depending of the case, phi_0 might be not the best candidate
        // and the distance on the ray can be negative
        let theta_u = crate::math::acos_fast(
            DirectionCone::subtended_directions(&self.aabb, &closest).cos_theta,
        );
        let theta_p = (theta_min - self.theta_o - theta_u).max(0.0);
        if theta_p >= self.theta_e {
            0.0
        } else {
            // Axis vector projected inside equiangular plane
            // But parametrized in local coordinates
            (self.phi * theta_p.cos() / d).max(0.0)
        }
    }

    pub fn importance_point(&self, p: &Point3<f32>, n: Option<&Vector3<f32>>) -> f32 {
        let p_vec = p.to_vec();

        // Don't let d2 get too small if p is inside the bounds.
        let pc = self.aabb.center();
        let d2 = (p_vec - pc).magnitude2().max(EPSILON_ATS);

        let wi = (p_vec - pc).normalize();

        let mut cos_theta = self.w.dot(wi);
        if self.two_sided {
            cos_theta = cos_theta.abs();
        }

        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();

        // Define sine and cosine clamped subtraction lambdas
        // cos(max(0, a-b))
        let cos_sub_clamped =
            |sin_theta_a: f32, cos_theta_a: f32, sin_theta_b: f32, cos_theta_b: f32| -> f32 {
                if cos_theta_a > cos_theta_b {
                    1.0
                } else {
                    cos_theta_a * cos_theta_b + sin_theta_a * sin_theta_b
                }
            };
        // sin(max(0, a-b))
        let sin_sub_clamped =
            |sin_theta_a: f32, cos_theta_a: f32, sin_theta_b: f32, cos_theta_b: f32| -> f32 {
                if cos_theta_a > cos_theta_b {
                    1.0
                } else {
                    sin_theta_a * cos_theta_b - cos_theta_a * sin_theta_b
                }
            };

        // Compute $\cos \theta_\roman{u}$ for _intr_
        let cos_theta_u = DirectionCone::subtended_directions(&self.aabb, p).cos_theta;
        let sin_theta_u = (1.0 - cos_theta_u * cos_theta_u).max(0.0).sqrt();

        // Compute $\cos \theta_\roman{p}$ for _intr_ and test against $\cos
        // \theta_\roman{e}$
        // cos(theta_p). Compute in two steps
        let cos_theta_x = cos_sub_clamped(
            sin_theta,
            cos_theta,
            (1.0 - self.cos_theta_o * self.cos_theta_o).max(0.0).sqrt(),
            self.cos_theta_o,
        );
        let sin_theta_x = sin_sub_clamped(
            sin_theta,
            cos_theta,
            (1.0 - self.cos_theta_o * self.cos_theta_o).max(0.0).sqrt(),
            self.cos_theta_o,
        );
        let cos_theta_p = cos_sub_clamped(sin_theta_x, cos_theta_x, sin_theta_u, cos_theta_u);
        if cos_theta_p <= self.cos_theta_e {
            return 0.0;
        }

        let mut imp = self.phi * cos_theta_p / d2;

        // Account for $\cos \theta_\roman{i}$ in importance at surfaces
        if let Some(n) = n {
            // cos(thetap_i) = cos(max(0, theta_i - theta_u))
            // cos (a-b) = cos a cos b + sin a sin b
            let cos_theta_i = wi.dot(*n).abs();
            let sin_theta_i = (1.0 - cos_theta_i * cos_theta_i).max(0.0).sqrt();
            let cos_thetap_i = cos_sub_clamped(sin_theta_i, cos_theta_i, sin_theta_u, cos_theta_u);
            imp *= cos_thetap_i;
        }

        imp.max(0.0)
    }
}

pub struct LightProxy {
    emitter_id: usize,
    primitive_idx: usize,
    bounds: LightBounds,
}

pub struct LightBVHNode {
    // Tree structure
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub parent: Option<usize>,
    // Bound of the current node
    pub bounds: LightBounds,
    pub light: Option<usize>, // Only for leaf nodes
}
impl LightBVHNode {
    pub fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }
}
pub struct LightSamplerATS {
    // Root of the tree
    pub root: Option<usize>,
    // Nodes and lights
    pub nodes: Vec<LightBVHNode>,
    pub lights: Vec<LightProxy>,
    // To map light query to node
    // to compute PDF
    pub query_to_nodes: HashMap<(usize, usize), usize>,
}

fn max_component(v: Vector3<f32>) -> f32 {
    v.x.max(v.y).max(v.z)
}

fn build_bvh(
    nodes: &mut Vec<LightBVHNode>,
    query_to_nodes: &mut HashMap<(usize, usize), usize>,
    index: usize,
    lights: &mut [LightProxy],
) -> usize {
    match &lights {
        [] => unimplemented!(),
        [el] => {
            nodes.push(LightBVHNode {
                left: None,
                right: None,
                parent: None,
                bounds: el.bounds.clone(),
                light: Some(index),
            });
            // Update the maps
            query_to_nodes.insert((el.emitter_id, el.primitive_idx), nodes.len() - 1);
            nodes.len() - 1
        }
        _ => {
            let (bounds, centroid_bounds) = {
                let mut bounds = AABB::default();
                let mut centroid_bounds = AABB::default();
                for l in lights.iter() {
                    bounds = bounds.union_aabb(&l.bounds.aabb);
                    centroid_bounds = centroid_bounds.union_vec(&l.bounds.aabb.center());
                }
                (bounds, centroid_bounds)
            };

            let mut min_cost = std::f32::MAX;
            let mut min_cost_bucket = -1;
            let mut min_cost_dim = -1;

            const NBUCKETS: usize = 12;
            for dim in 0..3 {
                // Nothing, skip
                if centroid_bounds.p_max[dim] == centroid_bounds.p_min[dim] {
                    continue;
                }

                // Compute bounds for each buckets
                // by discretizing the space
                let buckets_bounds = {
                    let mut b = vec![LightBounds::default(); NBUCKETS];
                    for l in lights.iter() {
                        let pc = l.bounds.aabb.center();
                        let i = (NBUCKETS as f32 * centroid_bounds.offset(&pc)[dim]) as usize;
                        let i = i.min(NBUCKETS - 1);
                        b[i] = LightBounds::union(&b[i], &l.bounds);
                    }
                    b
                };

                let cost = {
                    let mut c = [0.0; NBUCKETS - 1];
                    for i in 0..(NBUCKETS - 1) {
                        let b0 = {
                            let mut b = LightBounds::default();
                            for j in 0..(i + 1) {
                                b = LightBounds::union(&b, &buckets_bounds[j]);
                            }
                            b
                        };
                        let b1 = {
                            let mut b = LightBounds::default();
                            for j in (i + 1)..NBUCKETS {
                                b = LightBounds::union(&b, &buckets_bounds[j]);
                            }
                            b
                        };

                        let momega = |b: &LightBounds| -> f32 {
                            let theta_w = (b.theta_o + b.theta_e).min(std::f32::consts::PI);
                            2.0 * std::f32::consts::PI * (1.0 - b.theta_o.cos())
                                + std::f32::consts::FRAC_PI_2
                                    * (2.0 * theta_w * b.theta_o.sin()
                                        - (b.theta_o - 2.0 * theta_w).cos()
                                        - 2.0 * b.theta_o * b.theta_o.sin()
                                        + b.theta_o.cos())
                        };

                        let kr = max_component(bounds.size()) / bounds.size()[dim];
                        c[i] = kr
                            * (b0.phi * momega(&b0) * b0.aabb.surface_area()
                                + b1.phi * momega(&b1) * b1.aabb.surface_area())
                    }

                    c
                };

                for (i, c) in cost.iter().enumerate() {
                    if *c > 0.0 && *c < min_cost {
                        // Find new candidate
                        min_cost = *c;
                        min_cost_bucket = i as i32;
                        min_cost_dim = dim as i32;
                    }
                }
            }

            let mid = {
                if min_cost_dim == -1 {
                    // No bucket found, just split in the mid
                    lights.len() / 2
                } else {
                    let min_cost_dim = min_cost_dim as usize;
                    let min_cost_bucket = min_cost_bucket as usize;

                    itertools::partition(lights.iter_mut(), |l| {
                        let pc = l.bounds.aabb.center();
                        let i =
                            (NBUCKETS as f32 * centroid_bounds.offset(&pc)[min_cost_dim]) as usize;
                        let i = i.min(NBUCKETS - 1);
                        i <= min_cost_bucket
                    })
                }
            };

            // Recursive call
            let (left_lights, right_lights) = lights.split_at_mut(mid);
            let left = build_bvh(nodes, query_to_nodes, index, left_lights);
            let right = build_bvh(nodes, query_to_nodes, index + mid, right_lights);

            // Create new node with combine left and right
            nodes.push(LightBVHNode {
                left: Some(left),
                right: Some(right),
                parent: None,
                bounds: LightBounds::union(&nodes[left].bounds, &nodes[right].bounds),
                light: None, // It is not an leaf node
            });

            // Update childs
            let id = nodes.len() - 1;
            nodes[left].parent = Some(id);
            nodes[right].parent = Some(id);

            id
        }
    }
}

impl LightSamplerATS {
    fn new(emitters: &Vec<Arc<dyn Emitter>>) -> Option<LightSamplerATS> {
        // For now we do not handle both light type
        for e in emitters {
            assert!(e.is_surface());
        }
        let mut lights = emitters
            .iter()
            .enumerate()
            .map(|(i, e)| e.convert_light_proxy(i))
            .flatten()
            .collect::<Vec<_>>();

        if lights.is_empty() {
            None
        } else {
            let mut nodes = Vec::with_capacity(lights.len());
            let mut query_to_nodes = HashMap::with_capacity(lights.len());
            let id = build_bvh(&mut nodes, &mut query_to_nodes, 0, &mut lights[..]); // Build nodes recursively
            dbg!(nodes.len());
            dbg!(lights.len());
            Some(LightSamplerATS {
                root: Some(id),
                nodes,
                lights,
                query_to_nodes,
            })
        }
    }

    fn pdf<F>(&self, id_emitter: usize, id_primitive: usize, importance: F) -> f32
    where
        F: Fn(&LightBounds) -> f32,
    {
        let mut id = *self
            .query_to_nodes
            .get(&(id_emitter, id_primitive))
            .unwrap();
        let mut node = &self.nodes[id];

        let mut pdf = 1.0;
        while let Some(id_parent) = node.parent {
            // Update node
            node = &self.nodes[id_parent];

            // Compute importance of the two childs
            let imp_left = importance(&self.nodes[node.left.unwrap()].bounds);
            let imp_right = importance(&self.nodes[node.right.unwrap()].bounds);
            let imp_total = imp_left + imp_right;

            // Compute prob
            let prob_left = if imp_left == 0.0 && imp_right == 0.0 {
                // return None;
                0.5
            } else {
                imp_left / imp_total
            };

            // let prob_left = imp_left / imp_total;
            if node.left.unwrap() == id {
                pdf *= prob_left;
            } else {
                pdf *= 1.0 - prob_left;
            }

            // Update ID
            id = id_parent;
        }

        pdf
    }

    fn sample<F>(&self, mut r: f32, importance: F) -> Option<(&LightProxy, f32)>
    where
        F: Fn(&LightBounds) -> f32,
    {
        assert!(self.root.is_some());

        let mut pdf_sel = 1.0;
        let mut node_index = self.root.unwrap();
        loop {
            let node = &self.nodes[node_index];
            if node.is_leaf() {
                return Some((&self.lights[node.light.unwrap()], pdf_sel));
            } else {
                let imp_left = importance(&self.nodes[node.left.unwrap()].bounds);
                let imp_right = importance(&self.nodes[node.right.unwrap()].bounds);
                let imp_total = imp_left + imp_right;

                let prob_left = if imp_left == 0.0 && imp_right == 0.0 {
                    0.5
                } else {
                    imp_left / imp_total
                };

                if r < prob_left {
                    // Left choosed
                    // r = rand::random();
                    r = r / prob_left;
                    node_index = node.left.unwrap();
                    pdf_sel *= prob_left;
                } else {
                    // Right choosed
                    // r = rand::random();
                    r = (r - prob_left) / (1.0 - prob_left);
                    node_index = node.right.unwrap();
                    pdf_sel *= 1.0 - prob_left;
                }
            }
        }
    }

    fn sample_split<F, F2>(
        &self,
        mut r: f32,
        importance: F,
        variance_g: F2,
        splitting_factor: f32,
        sampler: &mut dyn Sampler,
    ) -> Vec<(&LightProxy, f32)>
    where
        F: Fn(&LightBounds) -> f32,
        F2: Fn(&LightBounds) -> (f32, f32),
    {
        assert!(self.root.is_some());

        let mut pdf_sel = 1.0;
        let mut node_index = self.root.unwrap();
        let mut node_selected = Vec::new();
        let mut node_queued: Vec<(usize, f32, f32)> = Vec::new();

        loop {
            let node = &self.nodes[node_index];
            if node.is_leaf() {
                if importance(&self.lights[node.light.unwrap()].bounds) > 0.0 {
                    // Add the node
                    node_selected.push((&self.lights[node.light.unwrap()], pdf_sel));
                    if node_queued.is_empty() {
                        return node_selected;
                    }

                    // Go to the next node to treat
                    let res = node_queued.pop().unwrap();
                    node_index = res.0;
                    pdf_sel = res.1;
                    r = res.2;
                }
            } else {
                // Compute split measure
                let ee = node.bounds.phi;
                let ve = (node.bounds.phi_sqr / node.bounds.number_lights as f32)
                    - (node.bounds.phi / node.bounds.number_lights as f32).powi(2);
                let (eg, vg) = variance_g(&node.bounds);
                let split_measure = ve * vg + ve * eg.powi(2) + ee.powi(2) * vg;
                let split_measure = (1.0
                    / (1.0
                        + (node.bounds.number_lights * node.bounds.number_lights) as f32
                            * split_measure))
                    .powf(0.25);

                if split_measure < splitting_factor {
                    node_queued.push((node.left.unwrap(), pdf_sel, sampler.next()));
                    node_index = node.right.unwrap();
                } else {
                    let imp_left = importance(&self.nodes[node.left.unwrap()].bounds);
                    let imp_right = importance(&self.nodes[node.right.unwrap()].bounds);
                    let imp_total = imp_left + imp_right;

                    let prob_left = if imp_left == 0.0 && imp_right == 0.0 {
                        if node_queued.is_empty() {
                            return node_selected;
                        }

                        let res = node_queued.pop().unwrap();
                        node_index = res.0;
                        pdf_sel = res.1;
                        r = res.2;
                        continue;
                    } else {
                        imp_left / imp_total
                    };

                    if r < prob_left {
                        // Left choosed
                        // r = rand::random();
                        r = r / prob_left;
                        node_index = node.left.unwrap();
                        pdf_sel *= prob_left;
                    } else {
                        // Right choosed
                        // r = rand::random();
                        r = (r - prob_left) / (1.0 - prob_left);
                        node_index = node.right.unwrap();
                        pdf_sel *= 1.0 - prob_left;
                    }
                }
            }
        }
    }
}

// This is the emitter sampler for now
pub struct EmitterSampler {
    pub emitters: Vec<Arc<dyn Emitter>>,
    pub emitters_cdf: Distribution1D,
    pub ats: Option<LightSamplerATS>,
}

// TODO: See if there is solution in stable rust
// Need to transmute due to fat pointers
fn get_addr(emitter: &dyn Emitter) -> usize {
    let emitter_addr: [usize; 2] = unsafe { std::mem::transmute(emitter) };
    emitter_addr[0]
}

impl EmitterSampler {
    pub fn build_ats(&mut self) {
        info!("Building ATS...");
        self.ats = Some(LightSamplerATS::new(&self.emitters).unwrap());
    }

    pub fn pdf(&self, emitter: &dyn Emitter) -> f32 {
        let emitter_addr = get_addr(emitter);
        for (i, e) in self.emitters.iter().enumerate() {
            let other_addr = get_addr(e.as_ref());
            if emitter_addr == other_addr {
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

    pub fn direct_pdf_ray(
        &self,
        emitter: &dyn Emitter,
        light_sampling: &LightSamplingPDF,
        ray: &Ray,
        max_dist: Option<f32>,
        id_primitive: Option<usize>,
    ) -> PDF {
        match &self.ats {
            None => emitter.direct_pdf(light_sampling) * self.pdf(emitter),
            Some(ats) => {
                // Compute id emitter
                let id_emitter = {
                    let emitter_addr = get_addr(emitter);
                    let mut id_emitter = None;
                    for (i, e) in self.emitters.iter().enumerate() {
                        let other_addr = get_addr(e.as_ref());
                        if emitter_addr == other_addr {
                            id_emitter = Some(i);
                            break;
                        }
                    }
                    if id_emitter.is_none() {
                        warn!("PDF emitter without intersecting an emitter");
                        return PDF::SolidAngle(0.0);
                    }
                    id_emitter.unwrap()
                };
                let id_primitive = id_primitive.unwrap();

                // Pdf tri * pdf ATS
                let f = |bounds: &LightBounds| -> f32 { bounds.importance_ray(ray, max_dist) };
                emitter.direct_pdf_tri(light_sampling, id_primitive)
                    * ats.pdf(id_emitter, id_primitive, f)
            }
        }
    }

    pub fn direct_pdf(
        &self,
        emitter: &dyn Emitter,
        light_sampling: &LightSamplingPDF,
        n: Option<&Vector3<f32>>,
        id_primitive: Option<usize>,
    ) -> PDF {
        match &self.ats {
            None => emitter.direct_pdf(light_sampling) * self.pdf(emitter),
            Some(ats) => {
                // Compute id emitter
                let id_emitter = {
                    let emitter_addr = get_addr(emitter);
                    let mut id_emitter = None;
                    for (i, e) in self.emitters.iter().enumerate() {
                        let other_addr = get_addr(e.as_ref());
                        if emitter_addr == other_addr {
                            id_emitter = Some(i);
                            break;
                        }
                    }
                    if id_emitter.is_none() {
                        warn!("PDF emitter without intersecting an emitter");
                        return PDF::SolidAngle(0.0);
                    }
                    id_emitter.unwrap()
                };
                let id_primitive = id_primitive.unwrap();

                // Pdf tri * pdf ATS
                let f =
                    |bounds: &LightBounds| -> f32 { bounds.importance_point(&light_sampling.o, n) };
                emitter.direct_pdf_tri(light_sampling, id_primitive)
                    * ats.pdf(id_emitter, id_primitive, f)
            }
        }
    }

    pub fn sample_light(
        &self,
        p: &Point3<f32>,
        n: Option<&Vector3<f32>>,
        r_sel: f32,
        r: f32,
        uv: Point2<f32>,
    ) -> LightSampling {
        match &self.ats {
            None => {
                // Select the point on the light
                let (pdf_sel, emitter) = self.random_select_emitter(r_sel);
                let mut res = emitter.direct_sample(p, r, uv);
                res.weight /= pdf_sel;
                res.pdf = res.pdf * pdf_sel;
                res
            }
            Some(ats) => {
                let f = |bounds: &LightBounds| -> f32 { bounds.importance_point(p, n) };
                match ats.sample(r_sel, f) {
                    Some((light_info, pdf_sel)) => {
                        let mut res = self.emitters[light_info.emitter_id].direct_sample_tri(
                            p,
                            light_info.primitive_idx,
                            uv,
                        );
                        res.weight /= pdf_sel;
                        res.pdf = res.pdf * pdf_sel;
                        res
                    }
                    None => {
                        unimplemented!()
                    }
                }
            }
        }
    }
    pub fn random_select_emitter(&self, v: f32) -> (f32, &dyn Emitter) {
        let id_light = self.emitters_cdf.sample_discrete(v);
        (
            self.emitters_cdf.pdf(id_light),
            self.emitters[id_light].as_ref(),
        )
    }

    pub fn random_sample_emitter_position_point(
        &self,
        p: &Point3<f32>,
        n: Option<&Vector3<f32>>,
        v: f32,
        uv: Point2<f32>,
    ) -> Option<(&dyn Emitter, SampledPosition, Color)> {
        let f = |bounds: &LightBounds| -> f32 { bounds.importance_point(p, n) };
        match self.ats.as_ref().unwrap().sample(v, f) {
            Some((light_info, pdf_sel)) => {
                let emitter = &self.emitters[light_info.emitter_id];
                let (mut sampled_pos, w) =
                    emitter.sample_position_tri(light_info.primitive_idx, uv);
                sampled_pos.pdf = sampled_pos.pdf * pdf_sel;
                Some((emitter.as_ref(), sampled_pos, w / pdf_sel))
            }
            None => None,
        }
    }

    pub fn random_sample_emitter_position_ray_splitting(
        &self,
        ray: &Ray,
        max_dist: Option<f32>,
        v: f32,
        uv: Point2<f32>,
        splitting_factor: f32,
        sampler: &mut dyn Sampler,
    ) -> Vec<(&dyn Emitter, SampledPosition, Color)> {
        let f = |bounds: &LightBounds| -> f32 { bounds.importance_ray(ray, max_dist) };
        let variance_g = |bounds: &LightBounds| -> (f32, f32) {
            // Most far away point
            let b1 = (ray.o - bounds.bsphere.center).magnitude2();
            let b2 = match max_dist {
                Some(v) => {
                    let p = ray.o + ray.d * v.min(10.0);
                    (p - bounds.bsphere.center).magnitude2()
                }
                None => {
                    let p = ray.o + ray.d * 10.0;
                    (p - bounds.bsphere.center).magnitude2()
                }
            };
            let b = b1.max(b2);
            let b = if b < bounds.bsphere.radius.powi(2) {
                EPSILON_ATS
            } else {
                (b.sqrt() - bounds.bsphere.radius).max(EPSILON_ATS)
            };

            // Minimal distance bounding sphere
            let (_, a) = crate::math::closest_squared_distance_ray_point(
                ray,
                max_dist,
                &bounds.bsphere.center,
            );
            let a = if a < bounds.bsphere.radius.powi(2) {
                EPSILON_ATS
            } else {
                (a.sqrt() - bounds.bsphere.radius).max(EPSILON_ATS)
            };

            // Linear
            let eg = (b.ln() - a.ln()) / (b - a);
            let vg = 1.0 / (a * b);
            (eg, vg)
        };

        self.ats
            .as_ref()
            .unwrap()
            .sample_split(v, f, variance_g, splitting_factor, sampler)
            .into_iter()
            .map(|(light_info, pdf_sel)| {
                let emitter = &self.emitters[light_info.emitter_id];
                let (mut sampled_pos, w) =
                    emitter.sample_position_tri(light_info.primitive_idx, uv);
                sampled_pos.pdf = sampled_pos.pdf * pdf_sel;
                (emitter.as_ref(), sampled_pos, w / pdf_sel)
            })
            .collect()
    }

    pub fn random_sample_emitter_position_ray(
        &self,
        ray: &Ray,
        max_dist: Option<f32>,
        v: f32,
        uv: Point2<f32>,
    ) -> Option<(&dyn Emitter, SampledPosition, Color)> {
        let f = |bounds: &LightBounds| -> f32 { bounds.importance_ray(ray, max_dist) };
        match self.ats.as_ref().unwrap().sample(v, f) {
            Some((light_info, pdf_sel)) => {
                let emitter = &self.emitters[light_info.emitter_id];
                let (mut sampled_pos, w) =
                    emitter.sample_position_tri(light_info.primitive_idx, uv);
                sampled_pos.pdf = sampled_pos.pdf * pdf_sel;
                Some((emitter.as_ref(), sampled_pos, w / pdf_sel))
            }
            None => None,
        }
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
