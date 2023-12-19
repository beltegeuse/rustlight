use crate::structure::Ray;
use cgmath::*;
use std;

pub fn closest_squared_distance_ray_point(
    ray: &Ray,
    max_dist: Option<f32>,
    p: &Point3<f32>,
) -> (Point3<f32>, f32) {
    let diff = p - ray.o;

    // Compute t
    let t = ray.d.dot(diff);
    let t = if t > 0.0 {
        match max_dist {
            Some(max_dist) => {
                if max_dist >= t {
                    t
                } else {
                    max_dist
                }
            }
            None => t,
        }
    } else {
        0.0
    };

    let closest = ray.o + ray.d * t;
    (closest, (p - closest).magnitude2())
}

pub fn abs_vec(v: &Vector3<f32>) -> Vector3<f32> {
    Vector3::new(v.x.abs(), v.y.abs(), v.z.abs())
}

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

pub fn acos_fast(x: f32) -> f32 {
    // From LTC realtime slides
    let x1 = x.abs();
    let x2 = x1 * x1;
    let x3 = x2 * x1;
    let s = -0.2121144 * x1 + std::f32::consts::FRAC_PI_2;
    let s = 0.0742610 * x2 + s;
    let s = -0.0187293 * x3 + s;
    let s = (1.0 - x1).sqrt() * s;
    if x >= 0.0 {
        s
    } else {
        std::f32::consts::PI - s
    }
}

pub struct NewtonSolverResult {
    pub pos: f32,
    pub nb_iter: u16,
    pub early_quit: bool,
    pub div_zero: bool,
}

// From: https://github.com/rust-lang/libm/blob/master/src/math/scalbn.rs
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn scalbn(x: f64, mut n: i32) -> f64 {
    let x1p1023 = f64::from_bits(0x7fe0000000000000); // 0x1p1023 === 2 ^ 1023
    let x1p53 = f64::from_bits(0x4340000000000000); // 0x1p53 === 2 ^ 53
    let x1p_1022 = f64::from_bits(0x0010000000000000); // 0x1p-1022 === 2 ^ (-1022)

    let mut y = x;

    if n > 1023 {
        y *= x1p1023;
        n -= 1023;
        if n > 1023 {
            y *= x1p1023;
            n -= 1023;
            if n > 1023 {
                n = 1023;
            }
        }
    } else if n < -1022 {
        /* make sure final n < -53 to avoid double
        rounding in the subnormal range */
        y *= x1p_1022 * x1p53;
        n += 1022 - 53;
        if n < -1022 {
            y *= x1p_1022 * x1p53;
            n += 1022 - 53;
            if n < -1022 {
                n = -1022;
            }
        }
    }
    y * f64::from_bits(((0x3ff + n) as u64) << 52)
}

// Pos: initial guess
// [min, max]: the range (for bisection)
// max_iter, digits: precision
// f_derv and f: function used for newtow
pub fn newton_raphson_iterate<F, G>(
    pos: f32,
    mut min: f32,
    mut max: f32,
    max_iter: u16,
    digits: usize,
    y: f32,
    f_derv: F,
    f: G,
) -> NewtonSolverResult
where
    F: Fn(f32) -> f32,
    G: Fn(f32) -> f32,
{
    let mut res = NewtonSolverResult {
        pos,
        nb_iter: 0,
        early_quit: false,
        div_zero: false,
    };

    // Replace: ldexp(1.0, 1 - digits) from float_extras
    let factor = scalbn(1.0, (1 - digits) as i32) as f32;
    let mut delta = std::f32::MAX;
    let mut delta1 = std::f32::MAX;

    let mut count = max_iter;

    loop {
        // Normalized CDF
        let f0 = f(res.pos) - y;
        // Normalized PDF
        let f1 = f_derv(res.pos);

        let delta2 = delta1;
        delta1 = delta;
        count -= 1;
        if 0.0 == f0 {
            break; // Converged!
        }
        if f1 == 0.0 {
            res.div_zero = true;
            break;
        } else {
            delta = f0 / f1;
        }

        if (delta * 2.0).abs() > (delta2).abs() {
            // last two steps haven't converged, try bisection:
            delta = if delta > 0.0 {
                (res.pos - min) / 2.0
            } else {
                (res.pos - max) / 2.0
            };
        }
        let old_pos = res.pos;
        res.pos -= delta; // Update the position

        // Condition when we escape the domain
        // Just go inside the middle
        if res.pos <= min {
            delta = 0.5 * (old_pos - min);
            res.pos = old_pos - delta;
            if (res.pos == min) || (res.pos == max) {
                break;
            }
        } else if res.pos >= max {
            delta = 0.5 * (old_pos - max);
            res.pos = old_pos - delta;
            if (res.pos == min) || (res.pos == max) {
                break;
            }
        }

        // update brackets:
        if delta > 0.0 {
            max = old_pos;
        } else {
            min = old_pos;
        }

        // Count the number of iterations
        res.nb_iter += 1;
        if count == 0 || (res.pos * factor).abs() >= delta.abs() {
            break; // Finish (exceed number iter or delta is too small)
        }
    }
    res.early_quit = res.nb_iter < max_iter;
    res
}

#[derive(Debug)]
pub enum SolutionCubic {
    Reals3(f32, f32, f32),
    Reals2(f32, f32),
    Real1(f32),
}

// From: http://math.ivanovo.ac.ru/dalgebra/Khashin/poly/index.html
// Solve:  x^3 + a*x^2 + b*x + c = 0
pub fn solve_cubic(a: f32, b: f32, c: f32) -> SolutionCubic {
    let a2 = a * a;
    let q = (a2 - 3.0 * b) / 9.0;
    let r = (a * (2.0 * a2 - 9.0 * b) + 27.0 * c) / 54.0;
    // equation x^3 + q*x + r = 0
    let r2 = r * r;
    let q3 = q * q * q;
    if r2 <= (q3 + 1.0e-14) {
        let mut t = r / q3.sqrt();
        if t < (-1.0) {
            t = -1.0;
        }
        if t > 1.0 {
            t = 1.0;
        }
        t = (t).acos();

        let a = a / 3.0;
        let q = -2.0 * q.sqrt();
        let x_0 = q * (t / 3.0).cos() - a;
        let x_1 = q * ((t + std::f32::consts::PI * 2.0) / 3.0).cos() - a;
        let x_2 = q * ((t - std::f32::consts::PI * 2.0) / 3.0).cos() - a;
        SolutionCubic::Reals3(x_0, x_1, x_2)
    } else {
        let root3 = |mut x: f32| {
            // From:  http://prografix.narod.ru
            if x == 0.0 {
                // Early out
                return 0.0;
            } else if x < 0.0 {
                x = -x;
            }

            let mut s = 1.;
            while x < 1. {
                x *= 8.;
                s *= 0.5;
            }
            while x > 8. {
                x *= 0.125;
                s *= 2.;
            }
            let mut r = 1.5;
            r -= 1. / 3. * (r - x / (r * r));
            r -= 1. / 3. * (r - x / (r * r));
            r -= 1. / 3. * (r - x / (r * r));
            r -= 1. / 3. * (r - x / (r * r));
            r -= 1. / 3. * (r - x / (r * r));
            r -= 1. / 3. * (r - x / (r * r));
            r * s
        };

        //A =-pow(fabs(r)+sqrt(r2-q3),1./3);
        let mut a_prime = -root3(r.abs() + (r2 - q3).sqrt());
        if r < 0.0 {
            a_prime = -a_prime;
        }
        let b_prime = if a_prime == 0.0 { 0.0 } else { q / a_prime };

        let a = a / 3.0;
        let x_0 = (a_prime + b_prime) - a;
        let x_1 = -0.5 * (a_prime + b_prime) - a;
        let x_2 = 0.5 * (3.0f32).sqrt() * (a_prime - b_prime);
        if x_2 < 1e-14 {
            SolutionCubic::Reals2(x_0, x_1)
        } else {
            SolutionCubic::Real1(x_0)
        }
    }
}

pub fn solve_quadratic_no_opt(a: f32, b: f32, c: f32) -> f32 {
    if a.abs() < 0.0001 {
        return -c / b;
    }
    let rad = b * b - 4. * a * c;
    if rad < 0. {
        return 0.;
    }
    let rad = (rad).sqrt();
    let tmp = (-b - rad) / (2. * a);
    if tmp < 0. || tmp > 1. {
        (-b + rad) / (2. * a)
    } else {
        tmp
    }
}

pub fn solve_quadratic(a: f32, b: f32, c: f32) -> Option<(f32, f32)> {
    if a == 0.0 {
        if b != 0.0 {
            let v = -c / b;
            Some((v, v))
        } else {
            None
        }
    } else {
        let d = b * b - 4.0 * a * c;
        if d < 0.0 {
            return None;
        }
        let d_sqrt = d.sqrt();
        let tmp = if b < 0.0 {
            -0.5 * (b - d_sqrt)
        } else {
            -0.5 * (b + d_sqrt)
        };

        let x0 = tmp / a;
        let x1 = c / tmp;
        if x0 > x1 {
            Some((x1, x0))
        } else {
            Some((x0, x1))
        }
    }
}

/// Create an orthogonal basis by taking the normal vector
/// code based on Pixar paper.
#[derive(Clone, Debug)]
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

    pub fn valid(&self) -> bool {
        self.0.x.is_finite() && self.0.y.is_finite() && self.0.z.is_finite()
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
#[derive(Debug)]
pub struct Distribution1DConstruct {
    pub elements: Vec<f32>,
}

pub struct Distribution1D {
    pub cdf: Vec<f32>,
    pub func: Vec<f32>,
    pub func_int: f32,
}

impl Distribution1DConstruct {
    pub fn new(l: usize) -> Distribution1DConstruct {
        let elements = Vec::with_capacity(l);
        Distribution1DConstruct { elements }
    }

    pub fn add(&mut self, v: f32) {
        self.elements.push(v);
    }

    pub fn normalize(self) -> Distribution1D {
        assert!(self.elements.len() > 0);

        // Create the new CDF
        let mut cdf = Vec::with_capacity(self.elements.len() + 1);
        let mut cur = 0.0;
        for e in &self.elements {
            cdf.push(cur);
            cur += e / self.elements.len() as f32;
        }
        cdf.push(cur);

        // Normalize the cdf
        if cur != 0.0 {
            cdf.iter_mut().for_each(|x| *x /= cur);
        }
        *cdf.last_mut().unwrap() = 1.0;

        Distribution1D {
            cdf,
            func: self.elements,
            func_int: cur,
        }
    }
}

impl Distribution1D {
    /// Sample an element in a discrete manner
    /// The function return the index of such element
    pub fn sample_discrete(&self, v: f32) -> usize {
        assert!(v >= 0.0);
        assert!(v < 1.0);
        match self
            .cdf
            .binary_search_by(|probe| probe.partial_cmp(&v).unwrap())
        {
            Ok(x) => x,
            Err(x) => x - 1,
        }
    }

    /// Sample an element in a continous manner
    /// The function return the wrapped position in [0, usize]
    pub fn sample_continuous(&self, v: f32) -> f32 {
        let i = self.sample_discrete(v);

        // Remap the random number between [0, 1]
        let dv = {
            let dv = v - self.cdf[i];
            let pdf = self.pdf(i);
            if pdf > 0.0 {
                // Normally, if the PDF is zero, there is no change to pick this entry
                // However, this can happens due to float inaccuracies
                dv / pdf
            } else {
                dv
            }
        };

        i as f32 + dv
    }

    pub fn pdf(&self, i: usize) -> f32 {
        self.cdf[i + 1] - self.cdf[i]
    }

    pub fn total(&self) -> f32 {
        self.func_int * (self.cdf.len() - 1) as f32
    }
}

pub struct Distribution2D {
    pub marginal: Distribution1D,
    pub conditionals: Vec<Distribution1D>,
}

impl Distribution2D {
    pub fn from_bitmap(image: &crate::structure::Bitmap) -> Distribution2D {
        let size_x = image.size.x as usize;
        let size_y = image.size.y as usize;

        // Build conditionals
        let mut marginal = Distribution1DConstruct::new(size_y);
        let conditionals = (0..size_y)
            .map(|y| {
                let mut conditional = Distribution1DConstruct::new(size_x);
                for x in 0..size_x {
                    let p = image.pixel(Point2::new(x as u32, y as u32));
                    conditional.add(p.luminance());
                }

                let conditional = conditional.normalize();
                marginal.add(conditional.func_int);
                conditional
            })
            .collect::<Vec<_>>();

        // Build marginal
        let marginal = marginal.normalize();
        Distribution2D {
            marginal,
            conditionals,
        }
    }

    pub fn sample_continuous(&self, uv: Point2<f32>) -> Point2<f32> {
        let y = self.marginal.sample_continuous(uv.y);
        let x = self.conditionals[y as usize].sample_continuous(uv.x);
        Point2::new(x, y)
    }

    pub fn pdf(&self, i: Point2<usize>) -> f32 {
        self.conditionals[i.y].func[i.x] / self.marginal.func_int
    }
}

////////// From rs_pbrt
// Functions uses to avoid self interseciton
// However, when using embree, this approach seems not working properly.

// Use **unsafe**
/// [std::mem::transmute_copy][transmute_copy]
/// to convert *f32* to *u32*.
///
/// [transmute_copy]: https://doc.rust-lang.org/std/mem/fn.transmute_copy.html
pub fn float_to_bits(f: f32) -> u32 {
    // uint64_t ui;
    // memcpy(&ui, &f, sizeof(double));
    // return ui;
    let rui: u32;
    unsafe {
        let ui: u32 = std::mem::transmute_copy(&f);
        rui = ui;
    }
    rui
}

/// Use **unsafe**
/// [std::mem::transmute_copy][transmute_copy]
/// to convert *u32* to *f32*.
///
/// [transmute_copy]: https://doc.rust-lang.org/std/mem/fn.transmute_copy.html
pub fn bits_to_float(ui: u32) -> f32 {
    // float f;
    // memcpy(&f, &ui, sizeof(uint32_t));
    // return f;
    let rf: f32;
    unsafe {
        let f: f32 = std::mem::transmute_copy(&ui);
        rf = f;
    }
    rf
}

/// Bump a floating-point value up to the next greater representable
/// floating-point value.
pub fn next_float_up(v: f32) -> f32 {
    if v.is_infinite() && v > 0.0 {
        v
    } else {
        let new_v = if v == -0.0 { 0.0 } else { v };
        let mut ui: u32 = float_to_bits(new_v);
        if new_v >= 0.0 {
            ui += 1;
        } else {
            ui -= 1;
        }
        bits_to_float(ui)
    }
}

/// Bump a floating-point value down to the next smaller representable
/// floating-point value.
pub fn next_float_down(v: f32) -> f32 {
    if v.is_infinite() && v < 0.0 {
        v
    } else {
        let new_v = if v == 0.0 { -0.0 } else { v };
        let mut ui: u32 = float_to_bits(new_v);
        if new_v > 0.0 {
            ui -= 1;
        } else {
            ui += 1;
        }
        bits_to_float(ui)
    }
}
