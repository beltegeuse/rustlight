use std::cmp;
use std::u32;
use rayon::prelude::*;
use cgmath::*;
use image::*;
use embree;
use embree::rtcore::Intersection;
use rand;
use rustlight::structure::{Color, Ray};
use rustlight::camera::Camera;

pub struct Scene<'a> {
    // Camera parameters
    pub camera: Camera,
    // Geometry information
    pub embree: &'a embree::rtcore::Scene<'a>,
    pub bsdf: Vec<Color>, // per each objects
    // Parameters
    pub nb_samples: u32,
}

impl<'a> Scene<'a> {
    pub fn trace(&self, ray: &Ray) -> Option<Intersection> {
        let embree_ray = embree::rtcore::Ray::new(&ray.o, &ray.d);
        if let Some(inter) = self.embree.intersect(embree_ray) {
            Some(inter)
        } else {
            None
        }
    }
}

pub struct ImageBlock {
    pub pos: Point2<u32>,
    pub size: Vector2<u32>,
    pub pixels: Vec<Color>,
}

impl ImageBlock {
    pub fn new(pos: Point2<u32>, size: Vector2<u32>) -> ImageBlock {
        ImageBlock {
            pos: pos,
            size: size,
            pixels: vec![Color { r: 0.0, g: 0.0, b: 0.0 };
                         (size.x * size.y) as usize],
        }
    }

    pub fn accum(&mut self, p: Point2<u32>, f: &Color) {
        assert!(p.x < self.size.x);
        assert!(p.y < self.size.y);
        self.pixels[(p.y * self.size.y + p.x) as usize] += f;
    }

    pub fn get(&self, p: Point2<u32>) -> &Color {
        assert!(p.x < self.size.x);
        assert!(p.y < self.size.y);
        &self.pixels[(p.y * self.size.y + p.x) as usize]
    }

    pub fn weight(&mut self, f: f32) {
        assert!(f > 0.0);
        self.pixels.iter_mut().for_each(|v| v.mul(f));
    }
}

//////////
struct StepRangeInt(u32, u32, u32);

impl Iterator for StepRangeInt {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<u32> {
        if self.0 < self.1 {
            let v = self.0;
            self.0 = v + self.2;
            Some(v)
        } else {
            None
        }
    }
}

/////////////////////////
// Sampling
pub const INV_PI: f32 = 0.31830988618379067154;
pub const INV_2_PI: f32 = 0.15915494309189533577;
pub const PI_OVER_2: f32 = 1.57079632679489661923;
pub const PI_OVER_4: f32 = 0.78539816339744830961;
pub fn concentric_sample_disk(u: Point2<f32>) -> Point2<f32> {
    // map uniform random numbers to $[-1,1]^2$
    let u_offset: Point2<f32> = u * 2.0 as f32 - Vector2 { x: 1.0, y: 1.0 };
    // handle degeneracy at the origin
    if u_offset.x == 0.0 as f32 && u_offset.y == 0.0 as f32 {
        return Point2 { x : 0.0, y: 0.0 };
    }
    // apply concentric mapping to point
    let theta: f32;
    let r: f32;
    if u_offset.x.abs() > u_offset.y.abs() {
        r = u_offset.x;
        theta = PI_OVER_4 * (u_offset.y / u_offset.x);
    } else {
        r = u_offset.y;
        theta = PI_OVER_2 - PI_OVER_4 * (u_offset.x / u_offset.y);
    }
    Point2 {
        x: theta.cos(),
        y: theta.sin(),
    } * r
}

pub fn cosine_sample_hemisphere(u: Point2<f32>) -> Vector3<f32> {
    let d: Point2<f32> = concentric_sample_disk(u);
    let z: f32 = (0.0 as f32)
        .max(1.0 as f32 - d.x * d.x - d.y * d.y)
        .sqrt();
    Vector3 {
        x: d.x,
        y: d.y,
        z: z,
    }
}

pub fn basis(n : Vector3<f32>) -> Matrix3<f32> {
    let b1: Vector3<f32>;
    let b2: Vector3<f32>;
    if n.z<0.0 {
        let a = 1.0 / (1.0 - n.z);
        let b = n.x * n.y * a;
        b1 = Vector3::new(1.0 - n.x * n.x * a, -b, n.x);
        b2 = Vector3::new(b, n.y * n.y*a - 1.0, -n.y);
    } else{
        let a = 1.0 / (1.0 + n.z);
        let b = -n.x * n.y * a;
        b1 = Vector3::new(1.0 - n.x * n.x * a, b, -n.x);
        b2 = Vector3::new(b, 1.0 - n.y * n.y * a, -n.y);
    }
    Matrix3 {
        x : b1,
        y : b2,
        z : n
    }
}

/////////////////////////
// Functions
fn render_ao((ix,iy): (u32,u32),scene: &Scene) -> Option<Color> {
    let pix = (ix as f32 + rand::random::<f32>(), iy as f32 + rand::random::<f32>());
    let ray = scene.camera.generate(pix);

    // Do the intersection for the first path
    let intersection = scene.trace(&ray);
    if intersection.is_none() {
        return None;
    }
    return Some(Color::one(1.0));
//    let intersection = intersection.unwrap();
//
//    // Compute an new direction
//    let mut n_g_normalized = intersection.n_g.normalize();
//    if n_g_normalized.dot(ray.d) < 0.0 {
//        n_g_normalized = -n_g_normalized;
//    }
//    let frame = basis(n_g_normalized);
//
//    let d_local = cosine_sample_hemisphere(Point2::new(rand::random::<f32>(),
//                                                 rand::random::<f32>()));
//    let d = frame * d_local;
//
//    let o = intersection.p + d * 0.001;
//    let mut embree_ray_new = embree::rtcore::Ray::new(&o,
//                                                  &d);
//    scene.embree.occluded(&mut embree_ray_new);
//    if embree_ray_new.hit() {
//        Some(Color::one(1.0))
//    } else {
//        None
//    }
}

pub fn render(scene: &Scene) -> DynamicImage {
    assert!(scene.nb_samples != 0);

    // The image that we will render
    let mut image = DynamicImage::new_rgb8(scene.camera.size().x,
                                           scene.camera.size().y);

    // Create rendering blocks
    let mut image_blocks: Vec<ImageBlock> = Vec::new();
    for ix in StepRangeInt(0, scene.camera.size().x, 16) {
        for iy in StepRangeInt(0, scene.camera.size().y, 16) {
            let mut block = ImageBlock::new(
                Point2 { x: ix, y: iy },
                Vector2 {
                    x: cmp::min(16, scene.camera.size().x - ix),
                    y: cmp::min(16, scene.camera.size().y - iy),
                });
            image_blocks.push(block);
        }
    }

    // Render the image blocks
    image_blocks.iter_mut().for_each(|im_block|
        {
            for ix in 0..im_block.size.x {
                for iy in 0..im_block.size.y {
                    for _ in 0..scene.nb_samples {
                        if let Some(c) = render_ao((ix + im_block.pos.x, iy + im_block.pos.y), scene) {
                            im_block.accum(Point2 { x: ix, y: iy }, &c);
                        }
                    }
                }
            }
            im_block.weight(1.0 / (scene.nb_samples as f32));
        }
    );

    // Fill the image
    for im_block in image_blocks.iter() {
        for ix in 0..im_block.size.x {
            for iy in 0..im_block.size.y {
                image.put_pixel(ix + im_block.pos.x, iy + im_block.pos.y,
                                im_block.get(Point2 { x: ix, y: iy }).to_rgba())
            }
        }
    }
    image
}