use std::cmp;
use std::u32;

use rayon::prelude::*;
use cgmath::*;
use image::*;
use embree;

use rustlight::structure::{Color, Ray};
use rustlight::geometry::{Intersection, Intersectable};
use rustlight::camera::Camera;

pub struct Scene<'a> {
    // Camera parameters
    pub camera: Camera,
    pub embree: &'a embree::rtcore::Scene<'a>,
    pub bsdf: Vec<Color>,
}

impl<'a> Scene<'a> {
    pub fn trace(&self, ray: &Ray) -> Option<Intersection> {
        let embree_ray = embree::rtcore::Ray::new(&ray.o, &ray.d);
        if let Some(inter) = self.embree.intersect(embree_ray) {
            let inter = Intersection::new(&self.bsdf[inter.geom_id as usize]);
            Some(inter)
        } else {
            None
        }
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
    image_blocks.par_iter_mut().for_each(|im_block|
        for ix in 0..im_block.size.x {
            for iy in 0..im_block.size.y {
                let pix = ((ix + im_block.pos.x) as f32 + 0.5,
                (iy + im_block.pos.y) as f32 + 0.5);
                let ray = scene.camera.generate(pix);
                // Do the intersection
                let intersection = scene.trace(&ray);
                match intersection {
                    Some(x) => im_block.accum(Point2 { x: ix, y: iy },
                                              x.bsdf),
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
                                im_block.get(Point2 { x: ix, y: iy }).to_rgba())
            }
        }
    }
    image
}