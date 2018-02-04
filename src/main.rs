// For computing the time
use std::time::Instant;
use std::cmp;
use std::ops::{AddAssign};

// Vector math library
extern crate cgmath;

use cgmath::*;

// For saving image (ldr)
extern crate image;

use image::*;

// For easy parallelism
extern crate rayon;

use rayon::prelude::*;

//////////////////
// Structures
#[derive(Clone, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl<'b> AddAssign<&'b Color> for Color {
    fn add_assign(&mut self, other: &'b Color) {
        self.r += other.r;
        self.g += other.g;
        self.b += other.b;
    }
}


pub struct Sphere {
    pub pos: Point3<f32>,
    pub radius: f32,
    pub color: Color,
}

pub struct Ray {
    pub o: Point3<f32>,
    pub d: Vector3<f32>,
}

pub struct Scene {
    // Camera parameters
    pub res: Point2<u32>,
    pub fov: f32,
    // Primitives
    pub spheres: Vec<Sphere>,
}

pub struct Intersection<'a> {
    pub distance: f32,
    pub object: &'a Sphere,

    // No methods allowed
    _secret: (),
}

impl<'a> Intersection<'a> {
    pub fn new<'b>(distance: f32, element: &'b Sphere) -> Intersection<'b> {
        if !distance.is_finite() {
            panic!("Intersection must have a finite distance.");
        }
        Intersection {
            distance: distance,
            object: element,
            _secret: (),
        }
    }
}

impl Color {
    pub fn to_rgba(&self) -> Rgba<u8> {
        Rgba::from_channels((self.r * 255.0) as u8,
                            (self.g * 255.0) as u8,
                            (self.b * 255.0) as u8,
                            255)
    }
}

impl Ray {
    pub fn generate(impos: Point2<u32>, scene: &Scene) -> Ray {
        assert!(scene.res.x > scene.res.y);
        let fov_adjustment = (scene.fov.to_radians() / 2.0).tan();
        let aspect_ratio = (scene.res.x as f32) / (scene.res.y as f32);

        let sensor_x = (((impos.x as f32 + 0.5) / scene.res.x as f32) * 2.0 - 1.0) * aspect_ratio * fov_adjustment;
        let sensor_y = (1.0 - ((impos.y as f32 + 0.5) / scene.res.y as f32) * 2.0) * fov_adjustment;

        Ray {
            o: Point3 { x: 0.0, y: 0.0, z: 0.0 },
            d: Vector3 { x: sensor_x, y: sensor_y, z: -1.0 }.normalize(),
        }
    }
}

impl Scene {
    pub fn trace(&self, ray: &Ray) -> Option<Intersection> {
        self.spheres
            .iter()
            .filter_map(|e| e.intersect(ray).map(|d| Intersection::new(d, e)))
            .min_by(|i1, i2| i1.distance.partial_cmp(&i2.distance).unwrap())
    }
}

///////////////////////
// Traits
pub trait Intersectable {
    fn intersect(&self, ray: &Ray) -> Option<f32>;
}

impl Intersectable for Sphere {
    fn intersect(&self, ray: &Ray) -> Option<f32> {
        let l = self.pos - ray.o;
        let adj = l.dot(ray.d);
        let d2 = l.dot(l) - (adj * adj);
        let radius2 = self.radius * self.radius;
        if d2 > radius2 {
            return None;
        }
        let thc = (radius2 - d2).sqrt();
        let t0 = adj - thc;
        let t1 = adj + thc;

        if t0 < 0.0 && t1 < 0.0 {
            return None;
        }

        let distance = if t0 < t1 { t0 } else { t1 };
        Some(distance)
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
            pixels: vec![Color { r: 0.0, g : 0.0, b : 0.0};
                         (size.x * size.y) as usize],
        }
    }

    pub fn accum(&mut self, p: Point2<u32>, f: &Color) {
        assert!(p.x < self.size.x);
        assert!(p.y < self.size.y);
        assert!(p.x >= 0);
        assert!(p.y >= 0);
        self.pixels[(p.y * self.size.y + p.x) as usize] += f;
    }

    pub fn get(&self, p: Point2<u32>) -> &Color {
        assert!(p.x < self.size.x);
        assert!(p.y < self.size.y);
        assert!(p.x >= 0);
        assert!(p.y >= 0);
        &self.pixels[(p.y * self.size.y + p.x) as usize]
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
// Functions
pub fn render(scene: &Scene) -> DynamicImage {
    // The image that we will render
    let mut image = DynamicImage::new_rgb8(scene.res.x, scene.res.y);

    // Create rendering blocks
    let mut imageBlocks: Vec<ImageBlock> = Vec::new();
    for ix in StepRangeInt(0,scene.res.x,16) {
        for iy in StepRangeInt(0,scene.res.y,16) {
            let mut block = ImageBlock::new(
                Point2 { x: ix, y: iy},
                Vector2 {
                    x: cmp::min(16, scene.res.x - ix),
                    y: cmp::min(16, scene.res.y - iy),
                });
            imageBlocks.push(block);
        }
    }

    // Render the image blocks
    imageBlocks.par_iter_mut().map(|imBlock|
        for ix in 0..imBlock.size.x {
            for iy in 0..imBlock.size.y {
                let ray = Ray::generate(Point2 {
                    x: ix + imBlock.pos.x,
                    y: iy + imBlock.pos.y,
                }, scene);
                // Do the intersection
                let intersection = scene.trace(&ray);
                match intersection {
                    Some(x) => imBlock.accum(Point2 { x : ix, y: iy},
                                             &x.object.color),
                    None => (),
                }
            }
        }
    );

    // Fill the image
    for imBlock in imageBlocks.iter_mut() {
        for ix in 0..imBlock.size.x {
            for iy in 0..imBlock.size.y {
                image.put_pixel(ix + imBlock.pos.x, iy + imBlock.pos.y,
                                imBlock.get(Point2{x: ix, y: iy}).to_rgba())
            }
        }
    }
    image
}

fn main() {
    // Create some geometries
    let s1 = Sphere {
        pos: Point3 {
            x: 0.0,
            y: 0.0,
            z: -5.0,
        },
        radius: 1.0,
        color: Color {
            r: 0.4,
            g: 1.0,
            b: 0.4,
        },
    };

    // Define a default scene
    let scene = Scene {
        res: Point2 { x: 800, y: 600 },
        fov: 90.0,
        spheres: vec![s1],
    };

    // Some tests?
    let start = Instant::now();

    // Generate the thread pool
    let pool = rayon::ThreadPool::new(rayon::Configuration::new().num_threads(8)).unwrap();


    let img: DynamicImage = pool.install(|| render(&scene));
    assert_eq!(scene.res.x, img.width());
    assert_eq!(scene.res.y, img.height());

    let elapsed = start.elapsed();
    println!("Elapsed: {} ms",
             (elapsed.as_secs() * 1_000) + (elapsed.subsec_nanos() / 1_000_000) as u64);

    // Save the image
    let ref mut fout = std::fs::File::create("test.png").unwrap();
    img.save(fout, image::PNG);
}
