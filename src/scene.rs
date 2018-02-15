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
pub struct Bitmap {
    pub pos: Point2<u32>,
    pub size: Vector2<u32>,
    pub pixels: Vec<Color>,
}

impl Bitmap {
    pub fn new(pos: Point2<u32>, size: Vector2<u32>) -> Bitmap {
        Bitmap {
            pos: pos,
            size: size,
            pixels: vec![Color { r: 0.0, g: 0.0, b: 0.0 };
                         (size.x * size.y) as usize],
        }
    }

    pub fn accum_bitmap(&mut self, o: &Bitmap) {
        for x in (0..o.size.x) {
            for y in (0..o.size.y) {
                let c_p = Point2::new(o.pos.x + x, o.pos.y + y);
                self.accum(c_p, o.get(Point2::new(x,y)));
            }
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
    pub fn render(&self) -> Bitmap {
        assert!(self.nb_samples != 0);

        // Create rendering blocks
        let mut image_blocks: Vec<Bitmap> = Vec::new();
        for ix in StepRangeInt::new(0, self.camera.size().x as usize, 16) {
            for iy in StepRangeInt::new(0, self.camera.size().y as usize, 16) {
                let mut block = Bitmap::new(
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
        let mut image = Bitmap::new(Point2::new(0,0), self.camera.size().clone());
        for im_block in image_blocks.iter() {
           image.accum_bitmap(&im_block);
        }
        image
    }
}