// Code from rs_pbrt

// std
use cgmath::{InnerSpace, Point2, Vector3};
use std::f32::consts::PI;

pub fn spherical_direction(sin_theta: f32, cos_theta: f32, phi: f32) -> Vector3<f32> {
    Vector3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta)
}

pub fn clamp_t<T>(val: T, low: T, high: T) -> T
where
    T: PartialOrd,
{
    let r: T;
    if val < low {
        r = low;
    } else if val > high {
        r = high;
    } else {
        r = val;
    }
    r
}

pub fn vec3_same_hemisphere_vec3(w: &Vector3<f32>, wp: &Vector3<f32>) -> bool {
    w.z * wp.z > 0.0 as f32
}

/// Utility function to calculate cosine via spherical coordinates.
pub fn cos_theta(w: &Vector3<f32>) -> f32 {
    w.z
}

/// Utility function to calculate the square cosine via spherical
/// coordinates.
pub fn cos_2_theta(w: &Vector3<f32>) -> f32 {
    w.z * w.z
}

/// Utility function to calculate the absolute value of the cosine via
/// spherical coordinates.
pub fn abs_cos_theta(w: &Vector3<f32>) -> f32 {
    w.z.abs()
}

/// Utility function to calculate the square sine via spherical
/// coordinates.
pub fn sin_2_theta(w: &Vector3<f32>) -> f32 {
    (0.0 as f32).max(1.0 as f32 - cos_2_theta(w))
}

/// Utility function to calculate sine via spherical coordinates.
pub fn sin_theta(w: &Vector3<f32>) -> f32 {
    sin_2_theta(w).sqrt()
}

/// Utility function to calculate the tangent via spherical
/// coordinates.
pub fn tan_theta(w: &Vector3<f32>) -> f32 {
    sin_theta(w) / cos_theta(w)
}

/// Utility function to calculate the square tangent via spherical
/// coordinates.
pub fn tan_2_theta(w: &Vector3<f32>) -> f32 {
    sin_2_theta(w) / cos_2_theta(w)
}

/// Utility function to calculate cosine via spherical coordinates.
pub fn cos_phi(w: &Vector3<f32>) -> f32 {
    let sin_theta: f32 = sin_theta(w);
    if sin_theta == 0.0 as f32 {
        1.0 as f32
    } else {
        clamp_t(w.x / sin_theta, -1.0, 1.0)
    }
}

/// Utility function to calculate sine via spherical coordinates.
pub fn sin_phi(w: &Vector3<f32>) -> f32 {
    let sin_theta: f32 = sin_theta(w);
    if sin_theta == 0.0 as f32 {
        0.0 as f32
    } else {
        clamp_t(w.y / sin_theta, -1.0, 1.0)
    }
}

/// Utility function to calculate square cosine via spherical coordinates.
pub fn cos_2_phi(w: &Vector3<f32>) -> f32 {
    cos_phi(w) * cos_phi(w)
}

/// Utility function to calculate square sine via spherical coordinates.
pub fn sin_2_phi(w: &Vector3<f32>) -> f32 {
    sin_phi(w) * sin_phi(w)
}

// see microfacet.h
pub trait MicrofacetDistribution {
    fn d(&self, wh: &Vector3<f32>) -> f32;
    fn lambda(&self, w: &Vector3<f32>) -> f32;
    fn g1(&self, w: &Vector3<f32>) -> f32 {
        1.0 as f32 / (1.0 as f32 + self.lambda(w))
    }
    fn g(&self, wo: &Vector3<f32>, wi: &Vector3<f32>) -> f32 {
        1.0 as f32 / (1.0 as f32 + self.lambda(wo) + self.lambda(wi))
    }
    fn pdf(&self, wo: &Vector3<f32>, wh: &Vector3<f32>) -> f32 {
        if self.get_sample_visible_area() {
            self.d(wh) * self.g1(wo) * wo.dot(*wh).abs() / abs_cos_theta(wo)
        } else {
            self.d(wh) * abs_cos_theta(wh)
        }
    }
    fn get_sample_visible_area(&self) -> bool;
}

pub struct TrowbridgeReitzDistribution {
    pub alpha_x: f32,
    pub alpha_y: f32,
    // inherited from class MicrofacetDistribution (see microfacet.h)
    pub sample_visible_area: bool,
}

impl TrowbridgeReitzDistribution {
    pub fn new(alpha_x: f32, alpha_y: f32, sample_visible_area: bool) -> Self {
        TrowbridgeReitzDistribution {
            alpha_x: alpha_x,
            alpha_y: alpha_y,
            sample_visible_area: sample_visible_area,
        }
    }
    /// Microfacet distribution function: In comparison to the
    /// Beckmann-Spizzichino model, Trowbridge-Reitz has higher tails - it
    /// falls off to zero more slowly for directions far from the surface
    /// normal.
    pub fn roughness_to_alpha(roughness: f32) -> f32 {
        let mut roughness = roughness;
        let limit: f32 = 1e-3 as f32;
        if limit > roughness {
            roughness = limit;
        }
        let x: f32 = roughness.ln(); // natural (base e) logarithm
        1.62142
            + 0.819955 * x
            + 0.1734 * x * x
            + 0.0171201 * x * x * x
            + 0.000640711 * x * x * x * x
    }
    pub fn sample_wh(&self, wo: &Vector3<f32>, u: &Point2<f32>) -> Vector3<f32> {
        let mut wh: Vector3<f32>;
        if !self.sample_visible_area {
            let cos_theta;
            let mut phi: f32 = (2.0 * PI) * u[1];
            if self.alpha_x == self.alpha_y {
                let tan_theta2: f32 = self.alpha_x * self.alpha_x * u[0] / (1.0 - u[0]);
                cos_theta = 1.0 / (1.0 + tan_theta2).sqrt();
            } else {
                phi = (self.alpha_y / self.alpha_x * (2.0 * PI * u[1] + 0.5 * PI).tan()).atan();
                if u[1] > 0.5 {
                    phi += PI;
                }
                let sin_phi: f32 = phi.sin();
                let cos_phi: f32 = phi.cos();
                let alphax2: f32 = self.alpha_x * self.alpha_x;
                let alphay2: f32 = self.alpha_y * self.alpha_y;
                let alpha2: f32 = 1.0 / (cos_phi * cos_phi / alphax2 + sin_phi * sin_phi / alphay2);
                let tan_theta2: f32 = alpha2 * u[0] / (1.0 - u[0]);
                cos_theta = 1.0 / (1.0 + tan_theta2).sqrt();
            }
            let sin_theta: f32 = (0.0 as f32).max(1.0 - cos_theta * cos_theta).sqrt();
            wh = spherical_direction(sin_theta, cos_theta, phi);
            if !vec3_same_hemisphere_vec3(wo, &wh) {
                wh = -wh;
            }
        } else {
            let flip: bool = wo.z < 0.0;
            if flip {
                wh = trowbridge_reitz_sample(&-(*wo), self.alpha_x, self.alpha_y, u[0], u[1]);
                wh = -wh;
            } else {
                wh = trowbridge_reitz_sample(wo, self.alpha_x, self.alpha_y, u[0], u[1]);
            }
        }
        wh
    }
}

impl MicrofacetDistribution for TrowbridgeReitzDistribution {
    fn d(&self, wh: &Vector3<f32>) -> f32 {
        let tan_2_theta: f32 = tan_2_theta(wh);
        if tan_2_theta.is_infinite() {
            return 0.0 as f32;
        }
        let cos_4_theta: f32 = cos_2_theta(wh) * cos_2_theta(wh);
        let e: f32 = (cos_2_phi(wh) / (self.alpha_x * self.alpha_x)
            + sin_2_phi(wh) / (self.alpha_y * self.alpha_y))
            * tan_2_theta;
        1.0 as f32
            / (PI * self.alpha_x * self.alpha_y * cos_4_theta * (1.0 as f32 + e) * (1.0 as f32 + e))
    }
    fn lambda(&self, w: &Vector3<f32>) -> f32 {
        let abs_tan_theta: f32 = tan_theta(w).abs();
        if abs_tan_theta.is_infinite() {
            return 0.0;
        }
        // compute _alpha_ for direction _w_
        let alpha: f32 = (cos_2_phi(w) * self.alpha_x * self.alpha_x
            + sin_2_phi(w) * self.alpha_y * self.alpha_y)
            .sqrt();
        let alpha_2_tan_2_theta: f32 = (alpha * abs_tan_theta) * (alpha * abs_tan_theta);
        (-1.0 as f32 + (1.0 as f32 + alpha_2_tan_2_theta).sqrt()) / 2.0 as f32
    }
    fn get_sample_visible_area(&self) -> bool {
        self.sample_visible_area
    }
}

fn trowbridge_reitz_sample_11(
    cos_theta: f32,
    u1: f32,
    u2: f32,
    slope_x: &mut f32,
    slope_y: &mut f32,
) {
    // special case (normal incidence)
    if cos_theta > 0.9999 {
        let r: f32 = (u1 / (1.0 - u1)).sqrt();
        let phi: f32 = 6.28318530718 * u2;
        *slope_x = r * phi.cos();
        *slope_y = r * phi.sin();
        return;
    }

    let sin_theta: f32 = (0.0 as f32).max(1.0 as f32 - cos_theta * cos_theta).sqrt();
    let tan_theta: f32 = sin_theta / cos_theta;
    let a: f32 = 1.0 / tan_theta;
    let g1: f32 = 2.0 / (1.0 + (1.0 + 1.0 / (a * a)).sqrt());

    // sample slope_x
    let a: f32 = 2.0 * u1 / g1 - 1.0;
    let mut tmp: f32 = 1.0 / (a * a - 1.0);
    if tmp > 1e10 {
        tmp = 1e10;
    }
    let b: f32 = tan_theta;
    let d: f32 = (b * b * tmp * tmp - (a * a - b * b) * tmp)
        .max(0.0 as f32)
        .sqrt();
    let slope_x_1: f32 = b * tmp - d;
    let slope_x_2: f32 = b * tmp + d;
    if a < 0.0 || slope_x_2 > 1.0 / tan_theta {
        *slope_x = slope_x_1;
    } else {
        *slope_x = slope_x_2;
    }

    // sample slope_y
    let s: f32;
    let new_u2: f32;
    if u2 > 0.5 {
        s = 1.0;
        new_u2 = 2.0 * (u2 - 0.5);
    } else {
        s = -1.0;
        new_u2 = 2.0 * (0.5 - u2);
    }
    let z: f32 = (new_u2 * (new_u2 * (new_u2 * 0.27385 - 0.73369) + 0.46341))
        / (new_u2 * (new_u2 * (new_u2 * 0.093073 + 0.309420) - 1.0) + 0.597999);
    *slope_y = s * z * (1.0 + *slope_x * *slope_x).sqrt();

    assert!(!(*slope_y).is_infinite());
    assert!(!(*slope_y).is_nan());
}

fn trowbridge_reitz_sample(
    wi: &Vector3<f32>,
    alpha_x: f32,
    alpha_y: f32,
    u1: f32,
    u2: f32,
) -> Vector3<f32> {
    // 1. stretch wi
    let wi_stretched: Vector3<f32> = Vector3::new(alpha_x * wi.x, alpha_y * wi.y, wi.z).normalize();

    // 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
    let mut slope_x: f32 = 0.0;
    let mut slope_y: f32 = 0.0;
    trowbridge_reitz_sample_11(cos_theta(&wi_stretched), u1, u2, &mut slope_x, &mut slope_y);

    // 3. rotate
    let tmp: f32 = cos_phi(&wi_stretched) * slope_x - sin_phi(&wi_stretched) * slope_y;
    slope_y = sin_phi(&wi_stretched) * slope_x + cos_phi(&wi_stretched) * slope_y;
    slope_x = tmp;

    // 4. unstretch
    slope_x = alpha_x * slope_x;
    slope_y = alpha_y * slope_y;

    // 5. compute normal
    Vector3::new(-slope_x, -slope_y, 1.0).normalize()
}
