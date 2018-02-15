use std::cmp;
use std::u32;
use std;

use rayon::prelude::*;
use cgmath::*;
use image::*;
use embree;
use serde_json;

// my includes
use structure::{Color, Ray};
use camera::{Camera, CameraParam};
use integrator::*;
use geometry;
use tools::StepRangeInt;

/// Image block
/// for easy paralelisation over the threads
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

/// Scene representation
pub struct Scene<'a> {
    /// Main camera
    pub camera: Camera,
    // Geometry information
    pub meshes: Vec<geometry::Mesh>,
    pub emitters: Vec<usize>,
    #[allow(dead_code)]
    embree_device: embree::rtcore::Device<'a>,
    embree_scene: embree::rtcore::Scene<'a>,
    // Parameters
    pub nb_samples: u32,
}

impl<'a> Scene<'a> {
    /// Take a json formatted string and an working directory
    /// and build the scene representation.
    pub fn new(data: String, wk: &std::path::Path) -> Result<Scene, String> {
        // Read json string
        let v: serde_json::Value = serde_json::from_str(&data).map_err(|e| e.to_string())?;

        // Allocate embree
        let mut device = embree::rtcore::Device::new();
        let mut scene_embree = device.new_scene(embree::rtcore::STATIC,
                                                embree::rtcore::INTERSECT1 | embree::rtcore::INTERPOLATE);

        // Read the object
        let obj_path = wk.join(v["meshes"].as_str().expect("impossible to read 'meshes' entry"));
        let mut meshes = geometry::load_obj(&mut scene_embree, obj_path.as_path()).expect("error during loading OBJ");

        // Build embree as we will not geometry for now
        println!("Build the acceleration structure");
        scene_embree.commit(); // Build

        // Update meshes informations
        //  - which are light?
        if let Some(emitters_json) = v.get("emitters") {
            for e in emitters_json.as_array().unwrap().iter() {
                let name: String = serde_json::from_value(e["mesh"].clone()).unwrap();
                let emission: Color = serde_json::from_value(e["emission"].clone()).unwrap();

                let mut found = false;
                for m in meshes.iter_mut() {
                    if m.name == name {
                        m.emission = emission.clone();
                        found = true;
                    }
                }

                if !found {
                    panic!("Not found {} in the obj list", name);
                } else {
                    println!("Ligth {:?} emission created", emission);
                }
            }
        }

        // Update the list of lights
        let mut emitters = vec![];
        for i in 0..meshes.len() {
            if !meshes[i].emission.is_zero() {
                emitters.push(i);
            }
        }

        // Read the camera config
        let camera_param: CameraParam = serde_json::from_value(v["camera"].clone()).unwrap();

        // Define a default scene
        Ok(Scene {
            camera: Camera::new(camera_param),
            embree_device: device,
            embree_scene: scene_embree,
            meshes,
            emitters,
            nb_samples: 128,
        })
    }

    /// Intersect and compute intersection informations
    pub fn trace(&self, ray: &Ray) -> Option<embree::rtcore::Intersection> {
        let embree_ray = embree::rtcore::Ray::new(&ray.o, &ray.d);
        if let Some(inter) = self.embree_scene.intersect(embree_ray) {
            Some(inter)
        } else {
            None
        }
    }

    /// Intersect the scene and return if we had an intersection or not
    pub fn hit(&self, ray: &Ray) -> bool {
        let mut embree_ray = embree::rtcore::Ray::new(&ray.o, &ray.d);
        self.embree_scene.occluded(&mut embree_ray);
        embree_ray.hit()
    }

    pub fn visible(&self, p0: &Point3<f32>, p1: &Point3<f32>) -> bool {
        let d = p1 - p0;
        let mut embree_ray = embree::rtcore::Ray::new(p0, &d);
        // FIXME: Use global constants
        embree_ray.tnear = 0.00001;
        embree_ray.tfar = 0.9999;

        self.embree_scene.occluded(&mut embree_ray);
        !embree_ray.hit()
    }

    /// Render the scene
    pub fn render(&self) -> DynamicImage {
        assert!(self.nb_samples != 0);

        // The image that we will render
        let mut image = DynamicImage::new_rgb8(self.camera.size().x,
                                               self.camera.size().y);

        // Create rendering blocks
        let mut image_blocks: Vec<ImageBlock> = Vec::new();
        for ix in StepRangeInt::new(0, self.camera.size().x as usize, 16) {
            for iy in StepRangeInt::new(0, self.camera.size().y as usize, 16) {
                let mut block = ImageBlock::new(
                    Point2 { x: ix as u32, y: iy as u32},
                    Vector2 {
                        x: cmp::min(16, self.camera.size().x - ix as u32),
                        y: cmp::min(16, self.camera.size().y - iy as u32),
                    });
                image_blocks.push(block);
            }
        }

        // Render the image blocks
        image_blocks.par_iter_mut().for_each(|im_block|
            {
                for ix in 0..im_block.size.x {
                    for iy in 0..im_block.size.y {
                        for _ in 0..self.nb_samples {
                            if let Some(c) = compute_direct((ix + im_block.pos.x, iy + im_block.pos.y), &self) {
                                im_block.accum(Point2 { x: ix, y: iy }, &c);
                            }
                        }
                    }
                }
                im_block.weight(1.0 / (self.nb_samples as f32));
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
}