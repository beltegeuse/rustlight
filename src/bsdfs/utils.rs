use crate::clamp;
use crate::structure::Color;
use cgmath::{InnerSpace, Vector3};

pub fn cos_theta(w: &Vector3<f32>) -> f32 {
    w.z
}
pub fn cos_2_theta(w: &Vector3<f32>) -> f32 {
    w.z * w.z
}
pub fn abs_cos_theta(w: &Vector3<f32>) -> f32 {
    w.z.abs()
}
pub fn sin_2_theta(w: &Vector3<f32>) -> f32 {
    (1.0 - cos_2_theta(w)).max(0.0)
}
pub fn sin_theta(w: &Vector3<f32>) -> f32 {
    sin_2_theta(w).sqrt()
}
pub fn tan_theta(w: &Vector3<f32>) -> f32 {
    sin_theta(w) / cos_theta(w)
}
pub fn tan2_theta(w: &Vector3<f32>) -> f32 {
    sin_2_theta(w) / cos_2_theta(w)
}
pub fn cos_phi(w: &Vector3<f32>) -> f32 {
    let sin_theta = sin_theta(w);
    if sin_theta == 0.0 {
        1.0
    } else {
        clamp(w.x / sin_theta, -1.0, 1.0)
    }
}
pub fn sin_phi(w: &Vector3<f32>) -> f32 {
    let sin_theta = sin_theta(w);
    if sin_theta == 0.0 {
        1.0
    } else {
        clamp(w.y / sin_theta, -1.0, 1.0)
    }
}
pub fn cos_2_phi(w: &Vector3<f32>) -> f32 {
    cos_phi(w) * cos_phi(w)
}
pub fn sin_2_phi(w: &Vector3<f32>) -> f32 {
    sin_phi(w) * sin_phi(w)
}

pub fn hypot2(a: f32, b: f32) -> f32 {
    if a.abs() > b.abs() {
        let r = b / a;
        a.abs() * (1.0 + r * r).sqrt()
    } else if b != 0.0 {
        let r = a / b;
        b.abs() * (1.0 + r * r).sqrt()
    } else {
        0.0
    }
}

pub fn reflect_vector(wo: Vector3<f32>, n: Vector3<f32>) -> Vector3<f32> {
    -(wo) + n * 2.0 * wo.dot(n)
}
pub fn check_reflection_condition(wi: &Vector3<f32>, wo: &Vector3<f32>) -> bool {
    (wi.z * wo.z - wi.x * wo.x - wi.y * wo.y - 1.0).abs() < 0.0001
}
pub fn check_direlectric_condition(
    wi: &Vector3<f32>,
    wo: &Vector3<f32>,
    eta: f32,
    cos_theta: f32,
) -> bool {
    let dot_p = -wi.x * wo.x * eta - wi.y * wo.y * eta - cos_theta.copysign(wi.z) * wo.z;
    (dot_p - 1.0).abs() < 0.0001
}
pub fn fresnel_conductor(cos_theta: f32, eta: Color, k: Color) -> Color {
    let cos_theta_2 = cos_theta * cos_theta;
    let sin_theta_2 = 1.0 - cos_theta_2;
    let sin_theta_4 = sin_theta_2 * sin_theta_2;

    let temp1 = eta * eta - k * k - Color::value(sin_theta_2);
    let a2pb2 = (temp1 * temp1 + k * k * eta * eta * 4.0).safe_sqrt();
    let a = ((a2pb2 + temp1) * 0.5).safe_sqrt();

    let term1 = a2pb2 + Color::value(cos_theta_2);
    let term2 = a * (2.0 * cos_theta_2);

    let rs2 = (term1 - term2) / (term1 + term2);

    let term3 = a2pb2 * cos_theta_2 + Color::value(sin_theta_4);
    let term4 = term2 * sin_theta_2;

    let rp2 = rs2 * (term3 - term4) / (term3 + term4);

    0.5 * (rp2 + rs2)
}

/// Return (fresnel, cosThetaT)
pub fn fresnel_dielectric(cos_theta_i_: f32, eta: f32) -> (f32, f32) {
    // Case of perfectly transparent object
    // we will just going though (and the fresnel is 0)
    if eta == 1.0 {
        return (0.0, -cos_theta_i_);
    }

    /* Using Snell's law, calculate the squared sine of the
    angle between the normal and the transmitted ray */
    let scale = if cos_theta_i_ > 0.0 { 1.0 / eta } else { eta };
    let cos_theta_t_sqr = 1.0 - (1.0 - cos_theta_i_ * cos_theta_i_) * (scale * scale);

    /* Check for total internal reflection */
    if cos_theta_t_sqr <= 0.0 {
        return (1.0, 0.0);
    }

    /* Find the absolute cosines of the incident/transmitted rays */
    let cos_theta_i = cos_theta_i_.abs();
    let cos_theta_t = cos_theta_t_sqr.sqrt();

    let rs = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);
    let rp = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);

    /* No polarization -- return the unpolarized reflectance */
    let cos_theta_t = if cos_theta_i_ > 0.0 {
        -cos_theta_t
    } else {
        cos_theta_t
    };
    (0.5 * (rs * rs + rp * rp), cos_theta_t)
}
