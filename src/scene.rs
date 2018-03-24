use BitmapTrait;
use camera::{Camera, CameraParam};
use cgmath::*;
use embree_rs;
use geometry;
use integrator::*;
use material::*;
use math::{Distribution1D, Distribution1DConstruct};
use pbr::ProgressBar;
// Other tools
use rayon::prelude::*;
use sampler;
use Scale;
use serde_json;
use std;
use std::cmp;
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::u32;
// my includes
use structure::*;
use tools::StepRangeInt;

/// Image block
/// for easy paralelisation over the thread
pub struct Bitmap<T: BitmapTrait> {
    pub pos: Point2<u32>,
    pub size: Vector2<u32>,
    pub pixels: Vec<T>,
}

impl<T: BitmapTrait> Bitmap<T> {
    pub fn new(pos: Point2<u32>, size: Vector2<u32>) -> Bitmap<T> {
        Bitmap {
            pos,
            size,
            pixels: vec![T::default();
                         (size.x * size.y) as usize],
        }
    }

    pub fn accumulate_bitmap(&mut self, o: &Bitmap<T>) {
        for x in 0..o.size.x {
            for y in 0..o.size.y {
                let c_p = Point2::new(o.pos.x + x, o.pos.y + y);
                self.accumulate(c_p, o.get(Point2::new(x, y)));
            }
        }
    }

    pub fn accumulate(&mut self, p: Point2<u32>, f: &T) {
        assert!(p.x < self.size.x);
        assert!(p.y < self.size.y);
        self.pixels[(p.y * self.size.y + p.x) as usize] += f.clone(); // FIXME: Not good for performance
    }

    pub fn accumulate_safe(&mut self, p: Point2<i32>, f: T) {
        if p.x >= 0
            && p.y >= 0
            && p.x < (self.size.x as i32)
            && p.y < (self.size.y as i32) {
            self.pixels[((p.y as u32) * self.size.y + p.x as u32) as usize] += f.clone(); // FIXME: Bad performance?
        }
    }

    pub fn get(&self, p: Point2<u32>) -> &T {
        assert!(p.x < self.size.x);
        assert!(p.y < self.size.y);
        &self.pixels[(p.y * self.size.y + p.x) as usize]
    }

    pub fn reset(&mut self) {
        self.pixels.iter_mut().for_each(|x| *x = T::default());
    }
}

impl<T: BitmapTrait> Scale<f32> for Bitmap<T> {
    fn scale(&mut self, f: f32) {
        assert!(f > 0.0);
        self.pixels.iter_mut().for_each(|v| v.scale(f));
    }
}

impl<T: BitmapTrait> Iterator for Bitmap<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        unimplemented!()
    }
}

/// Light sample representation
pub struct LightSampling<'a> {
    pub emitter: &'a geometry::Mesh,
    pub pdf: f32,
    pub p: Point3<f32>,
    pub n: Vector3<f32>,
    pub d: Vector3<f32>,
    pub weight: Color,
}

impl<'a> LightSampling<'a> {
    pub fn is_valid(&'a self) -> bool {
        self.pdf != 0.0
    }
}

pub struct LightSamplingPDF<'a> {
    pub mesh: &'a Arc<geometry::Mesh>,
    pub o: Point3<f32>,
    pub p: Point3<f32>,
    pub n: Vector3<f32>,
    pub dir: Vector3<f32>,
}

impl<'a> LightSamplingPDF<'a> {
    pub fn new(scene: &'a Scene,
               ray: &Ray,
               its: &embree_rs::ray::Intersection) -> LightSamplingPDF<'a> {
        LightSamplingPDF {
            mesh: &scene.meshes[its.geom_id as usize],
            o: ray.o,
            p: its.p,
            n: its.n_g,
            dir: ray.d,
        }
    }
}

/// Scene representation
pub struct Scene<'a> {
    /// Main camera
    pub camera: Camera,
    // Geometry information
    pub meshes: Vec<Arc<geometry::Mesh>>,
    pub emitters: Vec<Arc<geometry::Mesh>>,
    emitters_cdf: Distribution1D,
    #[allow(dead_code)]
    embree_device: embree_rs::scene::Device<'a>,
    embree_scene: embree_rs::scene::Scene<'a>,
}

impl<'a> Scene<'a> {
    /// Take a json formatted string and an working directory
    /// and build the scene representation.
    pub fn new(data: &str, wk: &std::path::Path) -> Result<Scene<'a>, Box<Error>> {
        // Read json string
        let v: serde_json::Value = serde_json::from_str(data)?;

        // Allocate embree
        let mut device = embree_rs::scene::Device::new();
        let mut scene_embree = device.new_scene(embree_rs::scene::SceneFlags::STATIC,
                                                embree_rs::scene::AlgorithmFlags::INTERSECT1);

        // Read the object
        let obj_path_str: String = v["meshes"].as_str().unwrap().to_string();
        let obj_path = wk.join(obj_path_str);
        let mut meshes = geometry::load_obj(&mut scene_embree, obj_path.as_path())?;

        // Build embree as we will not geometry for now
        println!("Build the acceleration structure");
        scene_embree.commit(); // Build

        // Update meshes information
        //  - which are light?
        if let Some(emitters_json) = v.get("emitters") {
            for e in emitters_json.as_array().unwrap() {
                let name: String = e["mesh"].as_str().unwrap().to_string();
                let emission: Color = serde_json::from_value(e["emission"].clone())?;
                // Get the set of matched meshes
                let mut matched_meshes = meshes.iter_mut().filter(|m| m.name == name).collect::<Vec<_>>();
                match matched_meshes.len() {
                    0 => panic!("Not found {} in the obj list", name),
                    1 => {
                        matched_meshes[0].emission = emission;
                    }
                    _ => panic!("Several {} in the obj list", name),
                };
            }
        }
        // - BSDF
        if let Some(bsdfs_json) = v.get("bsdfs") {
            for b in bsdfs_json.as_array().unwrap() {
                let name: String = serde_json::from_value(b["mesh"].clone())?;
                let new_bsdf_type: String = serde_json::from_value(b["type"].clone())?;
                let new_bsdf: Box<BSDF + Send + Sync> = match new_bsdf_type.as_ref() {
                    "phong" => Box::<BSDFPhong>::new(serde_json::from_value(b["data"].clone())?),
                    "diffuse" => Box::<BSDFDiffuse>::new(serde_json::from_value(b["data"].clone())?),
                    _ => panic!("Unknown BSDF type {}", new_bsdf_type),
                };

                let mut matched_meshes = meshes.iter_mut().filter(|m| m.name == name).collect::<Vec<_>>();
                match matched_meshes.len() {
                    0 => panic!("Not found {} in the obj list", name),
                    1 => {
                        matched_meshes[0].bsdf = new_bsdf;
                    }
                    _ => panic!("Several {} in the obj list", name),
                };
            }
        }

        // Transform the scene mesh from Box to Arc
        let meshes: Vec<Arc<geometry::Mesh>> = meshes.into_iter().map(|e| Arc::from(e)).collect();

        // Update the list of lights & construct the CDF
        let emitters = meshes.iter().filter(|m| !m.emission.is_zero())
            .map(|m| m.clone()).collect::<Vec<_>>();
        let emitters_cdf = {
            let mut cdf_construct = Distribution1DConstruct::new(emitters.len());
            emitters.iter().map(|e| e.flux()).for_each(|f| cdf_construct.add(f));
            cdf_construct.normalize()
        };

        // Read the camera config
        let camera_param: CameraParam = serde_json::from_value(v["camera"].clone()).unwrap();

        // Define a default scene
        Ok(Scene {
            camera: Camera::new(camera_param),
            embree_device: device,
            embree_scene: scene_embree,
            meshes,
            emitters,
            emitters_cdf,
        })
    }

    /// Intersect and compute intersection information
    pub fn trace(&self, ray: &Ray) -> Option<embree_rs::ray::Intersection> {
        let embree_ray = embree_rs::ray::Ray::new(
            &ray.o, &ray.d,
            ray.tnear, ray.tfar);
        self.embree_scene.intersect(embree_ray)
    }

    /// Intersect the scene and return if we had an intersection or not
    pub fn hit(&self, ray: &Ray) -> bool {
        let mut embree_ray = embree_rs::ray::Ray::new(
            &ray.o, &ray.d,
            ray.tnear, ray.tfar);
        self.embree_scene.occluded(&mut embree_ray);
        embree_ray.hit()
    }

    pub fn visible(&self, p0: &Point3<f32>, p1: &Point3<f32>) -> bool {
        let d = p1 - p0;
        let mut embree_ray = embree_rs::ray::Ray::new(
            p0, &d, 0.00001, 0.9999);
        self.embree_scene.occluded(&mut embree_ray);
        !embree_ray.hit()
    }

    pub fn direct_pdf(&self, light_sampling: LightSamplingPDF) -> f32 {
        let emitter_id = self.emitters.iter()
            .position(|m| Arc::ptr_eq(light_sampling.mesh, m)).unwrap();
        light_sampling.mesh.direct_pdf(light_sampling) * self.emitters_cdf.pdf(emitter_id)
    }
    pub fn sample_light(&self, p: &Point3<f32>, r_sel: f32, r: f32, uv: Point2<f32>) -> LightSampling {
        // Select the point on the light
        let (pdf_sel, emitter) = self.random_select_emitter(r_sel);
        let sampled_pos = emitter.sample(r, uv);

        // Compute the distance
        let mut d: Vector3<f32> = sampled_pos.p - p;
        let dist = d.magnitude();
        d /= dist;

        // Compute the geometry
        let cos_light = sampled_pos.n.dot(-d).max(0.0);
        let pdf = if cos_light == 0.0 { 0.0 } else { (pdf_sel * sampled_pos.pdf * dist * dist) / cos_light };
        let emission = if pdf == 0.0 { Color::zero() } else { emitter.emission / pdf };
        LightSampling {
            emitter,
            pdf,
            p: sampled_pos.p,
            n: sampled_pos.n,
            d,
            weight: emission,
        }
    }
    pub fn random_select_emitter(&self, v: f32) -> (f32, &geometry::Mesh) {
        let id_light = self.emitters_cdf.sample(v);
        (self.emitters_cdf.pdf(id_light), &self.emitters[id_light])
    }

    /// Render the scene
    pub fn render<T: BitmapTrait + Send>(&self,
                                         integrator: Box<Integrator<T> + Sync + Send>,
                                         nb_samples: u32) -> Bitmap<T> {
        assert_ne!(nb_samples, 0);

        // Create rendering blocks
        let mut image_blocks: Vec<Box<Bitmap<T>>> = Vec::new();
        for ix in StepRangeInt::new(0, self.camera.size().x as usize, 16) {
            for iy in StepRangeInt::new(0, self.camera.size().y as usize, 16) {
                let mut block = Bitmap::new(
                    Point2 { x: ix as u32, y: iy as u32 },
                    Vector2 {
                        x: cmp::min(16, self.camera.size().x - ix as u32),
                        y: cmp::min(16, self.camera.size().y - iy as u32),
                    });
                image_blocks.push(Box::new(block));
            }
        }

        // Render the image blocks
        let progress_bar = Mutex::new(ProgressBar::new(image_blocks.len() as u64));
        image_blocks.par_iter_mut().for_each(|im_block|
            {
                let mut sampler = sampler::IndependentSampler::default();
                for ix in 0..im_block.size.x {
                    for iy in 0..im_block.size.y {
                        for _ in 0..nb_samples {
                            let c = integrator.compute((ix + im_block.pos.x, iy + im_block.pos.y),
                                                       self, &mut sampler);
                            im_block.accumulate(Point2 { x: ix, y: iy }, &c);
                        }
                    }
                }
                im_block.scale(1.0 / (nb_samples as f32));

                {
                    progress_bar.lock().unwrap().inc();
                }
            }
        );

        // Fill the image
        let mut image = Bitmap::new(Point2::new(0, 0), *self.camera.size());
        for im_block in &image_blocks {
            image.accumulate_bitmap(im_block);
        }
        image
    }
}