use cgmath::{Vector3,InnerSpace};
use crate::structure::Color;

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

	let cos_theta_2 = cos_theta*cos_theta;
	let sin_theta_2 = 1.0-cos_theta_2;
	let sin_theta_4 = sin_theta_2*sin_theta_2;

	let temp1 = eta*eta - k*k - Color::value(sin_theta_2);
	let a2pb2 = (temp1*temp1 + k*k*eta*eta*4.0).safe_sqrt();
	let a     = ((a2pb2 + temp1) * 0.5).safe_sqrt();

	let term1 = a2pb2 + Color::value(cos_theta_2);
    let term2 = a*(2.0*cos_theta_2);

	let rs2 = (term1 - term2) / (term1 + term2);

	let term3 = a2pb2*cos_theta_2 + Color::value(sin_theta_4);
	let term4 = term2*sin_theta_2;

	let rp2 = rs2 * (term3 - term4) / (term3 + term4);

	0.5 * (rp2 + rs2)
}
