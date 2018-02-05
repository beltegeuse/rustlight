use std::cmp;
use rayon::prelude::*;
use cgmath::*;
use image::*;

use rustlight::structure::{Color,Ray};
use rustlight::geometry::{Sphere,Intersection,Intersectable};

pub struct Scene {
    // Camera parameters
    pub res: Point2<u32>,
    pub fov: f32,
    // Primitives
    pub spheres: Vec<Sphere>,
}

impl Scene {
    pub fn trace(&self, ray: &Ray) -> Option<Intersection> {
        self.spheres
            .iter()
            .filter_map(|e| e.intersect(ray).map(|d| Intersection::new(d, e)))
            .min_by(|i1, i2| i1.distance.partial_cmp(&i2.distance).unwrap())
    }
}

#[derive(Debug)]
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
        self.pixels[(p.y * self.size.y + p.x) as usize] += f;
    }

    pub fn get(&self, p: Point2<u32>) -> &Color {
        assert!(p.x < self.size.x);
        assert!(p.y < self.size.y);
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
    let mut image_blocks: Vec<ImageBlock> = Vec::new();
    for ix in StepRangeInt(0,scene.res.x,16) {
        for iy in StepRangeInt(0,scene.res.y,16) {
            let mut block = ImageBlock::new(
                Point2 { x: ix, y: iy},
                Vector2 {
                    x: cmp::min(16, scene.res.x - ix),
                    y: cmp::min(16, scene.res.y - iy),
                });
            image_blocks.push(block);
        }
    }

    // Render the image blocks
    image_blocks.par_iter_mut().for_each(|im_block|
        for ix in 0..im_block.size.x {
            for iy in 0..im_block.size.y {
                let ray = Ray::generate(Point2 {
                    x: ix + im_block.pos.x,
                    y: iy + im_block.pos.y,
                }, &scene.res, scene.fov);
                // Do the intersection
                let intersection = scene.trace(&ray);
                match intersection {
                    Some(x) => im_block.accum(Point2 { x : ix, y: iy},
                                              &x.object.color),
                    None => (),
                }
            }
        }
    );

    // Fill the image
    for im_block in image_blocks.iter() {
        for ix in 0..im_block.size.x {
            for iy in 0..im_block.size.y {
                image.put_pixel(ix + im_block.pos.x, iy + im_block.pos.y,
                                im_block.get(Point2{x: ix, y: iy}).to_rgba())
            }
        }
    }
    image
}