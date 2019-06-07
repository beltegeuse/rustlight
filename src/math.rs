use cgmath::*;
use std;

pub fn concentric_sample_disk(u: Point2<f32>) -> Point2<f32> {
    // map uniform random numbers to $[-1,1]^2$
    let u_offset: Point2<f32> = u * 2.0 as f32 - Vector2 { x: 1.0, y: 1.0 };
    // handle degeneracy at the origin
    if u_offset.x == 0.0 as f32 && u_offset.y == 0.0 as f32 {
        return Point2 { x: 0.0, y: 0.0 };
    }
    // apply concentric mapping to point
    let theta: f32;
    let r: f32;
    if u_offset.x.abs() > u_offset.y.abs() {
        r = u_offset.x;
        theta = std::f32::consts::FRAC_PI_4 * (u_offset.y / u_offset.x);
    } else {
        r = u_offset.y;
        theta =
            std::f32::consts::FRAC_PI_2 - std::f32::consts::FRAC_PI_4 * (u_offset.x / u_offset.y);
    }
    Point2 {
        x: theta.cos(),
        y: theta.sin(),
    } * r
}

pub fn cosine_sample_hemisphere(u: Point2<f32>) -> Vector3<f32> {
    let d: Point2<f32> = concentric_sample_disk(u);
    let z: f32 = (0.0 as f32).max(1.0 as f32 - d.x * d.x - d.y * d.y).sqrt();
    Vector3 { x: d.x, y: d.y, z }
}

pub fn sample_uniform_sphere(u: Point2<f32>) -> Vector3<f32> {
    let z = 1.0 - 2.0 * u.x;
    let r = (1.0 - z * z).max(0.0).sqrt();
    let phi = 2.0 * std::f32::consts::PI * u.y;
    Vector3::new(r * phi.cos(), r * phi.sin(), z)
}

/// Create an orthogonal basis by taking the normal vector
/// code based on Pixar paper.
#[derive(Clone)]
pub struct Frame(Matrix3<f32>);

impl Frame {
    pub fn new(n: Vector3<f32>) -> Frame {
        let sign = n.z.signum();
        let a = -1.0 / (sign + n.z);
        let b = n.x * n.y * a;
        Frame {
            0: Matrix3 {
                x: Vector3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x),
                y: Vector3::new(b, sign + n.y * n.y * a, -n.y),
                z: n,
            },
        }
    }

    pub fn to_world(&self, v: Vector3<f32>) -> Vector3<f32> {
        self.0.x * v.x + self.0.y * v.y + self.0.z * v.z
    }

    pub fn to_local(&self, v: Vector3<f32>) -> Vector3<f32> {
        Vector3::new(v.dot(self.0.x), v.dot(self.0.y), v.dot(self.0.z))
    }
}

/// Uniformly distributing samples over isosceles right triangles
/// actually works for any triangle.
pub fn uniform_sample_triangle(u: Point2<f32>) -> Point2<f32> {
    let su0: f32 = u.x.sqrt();
    Point2 {
        x: 1.0 as f32 - su0,
        y: u.y * su0,
    }
}

/// Create 1D distribution
pub struct Distribution1DConstruct {
    pub elements: Vec<f32>,
}

pub struct Distribution1D {
    pub cdf: Vec<f32>,
    pub normalization: f32,
}

impl Distribution1DConstruct {
    pub fn new(l: usize) -> Distribution1DConstruct {
        let elements = Vec::with_capacity(l);
        Distribution1DConstruct { elements }
    }

    pub fn add(&mut self, v: f32) {
        self.elements.push(v);
    }

    pub fn normalize(&mut self) -> Distribution1D {
        // Create the new CDF
        let mut cdf = Vec::with_capacity(self.elements.len() + 1);
        let mut cur = 0.0;
        for e in &self.elements {
            cdf.push(cur);
            cur += e;
        }
        cdf.push(cur);

        // Normalize the cdf
        cdf.iter_mut().for_each(|x| *x /= cur);

        Distribution1D {
            cdf,
            normalization: cur,
        }
    }
}

impl Distribution1D {
    pub fn sample(&self, v: f32) -> usize {
        assert!(v >= 0.0);
        assert!(v < 1.0);

        match self.cdf
            .binary_search_by(|probe| probe.partial_cmp(&v).unwrap())
        {
            Ok(x) => x,
            Err(x) => x - 1,
        }
    }

    pub fn pdf(&self, i: usize) -> f32 {
        assert!(i < self.cdf.len() - 1);
        self.cdf[i + 1] - self.cdf[i]
    }
}
