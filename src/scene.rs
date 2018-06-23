use bsdfs::*;
use camera::{Camera, CameraParam};
use cgmath::*;
use embree_rs;
use geometry;
use math::{Distribution1D, Distribution1DConstruct};
use serde_json;
use std;
use std::error::Error;
use std::sync::Arc;
use structure::*;

/// Light sample representation
pub struct LightSampling<'a> {
    pub emitter: &'a geometry::Mesh,
    pub pdf: PDF,
    pub p: Point3<f32>,
    pub n: Vector3<f32>,
    pub d: Vector3<f32>,
    pub weight: Color,
}

impl<'a> LightSampling<'a> {
    pub fn is_valid(&'a self) -> bool {
        !self.pdf.is_zero()
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
    pub fn new(ray: &Ray, its: &'a Intersection) -> LightSamplingPDF<'a> {
        LightSamplingPDF {
            mesh: &its.mesh,
            o: ray.o,
            p: its.p,
            n: its.n_g, // FIXME: Geometrical normal?
            dir: ray.d,
        }
    }
}

/// Scene representation
pub struct Scene {
    /// Main camera
    pub camera: Camera,
    // Geometry information
    pub meshes: Vec<Arc<geometry::Mesh>>,
    pub emitters: Vec<Arc<geometry::Mesh>>,
    emitters_cdf: Distribution1D,
    #[allow(dead_code)]
    embree_device: embree_rs::Device,
    embree_scene: embree_rs::Scene,
}

impl Scene {
    /// Take a json formatted string and an working directory
    /// and build the scene representation.
    pub fn new(data: &str, wk: &std::path::Path) -> Result<Scene, Box<Error>> {
        // Read json string
        let v: serde_json::Value = serde_json::from_str(data)?;

        // Allocate embree
        let mut device = embree_rs::Device::debug();
        let mut scene_embree = embree_rs::SceneConstruct::new(&device);

        // Read the object
        let obj_path_str: String = v["meshes"].as_str().unwrap().to_string();
        let obj_path = wk.join(obj_path_str);
        let mut meshes = geometry::load_obj(&device, &mut scene_embree, obj_path.as_path())?;

        // Build embree as we will not geometry for now
        info!("Build the acceleration structure");
        let scene_embree = scene_embree.commit()?;

        // Update meshes information
        //  - which are light?
        if let Some(emitters_json) = v.get("emitters") {
            for e in emitters_json.as_array().unwrap() {
                let name: String = e["mesh"].as_str().unwrap().to_string();
                let emission: Color = serde_json::from_value(e["emission"].clone())?;
                // Get the set of matched meshes
                let mut matched_meshes = meshes
                    .iter_mut()
                    .filter(|m| m.name == name)
                    .collect::<Vec<_>>();
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
                let new_bsdf = parse_bsdf(&b)?;
                let mut matched_meshes = meshes
                    .iter_mut()
                    .filter(|m| m.name == name)
                    .collect::<Vec<_>>();
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
        let emitters = meshes
            .iter()
            .filter(|m| !m.emission.is_zero())
            .map(|m| m.clone())
            .collect::<Vec<_>>();
        let emitters_cdf = {
            let mut cdf_construct = Distribution1DConstruct::new(emitters.len());
            emitters
                .iter()
                .map(|e| e.flux())
                .for_each(|f| cdf_construct.add(f));
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
    pub fn trace(&self, ray: &Ray) -> Option<Intersection> {
        match self.embree_scene.intersect(ray.to_embree()) {
            None => None,
            Some(its) => {
                let geom_id = its.geom_id as usize;
                Some(Intersection::new(its, -ray.d, &self.meshes[geom_id]))
            }
        }
    }

    pub fn visible(&self, p0: &Point3<f32>, p1: &Point3<f32>) -> bool {
        let d = p1 - p0;
        !self
            .embree_scene
            .occluded(embree_rs::Ray::new(*p0, d).near(0.00001).far(0.9999))
    }

    pub fn direct_pdf(&self, light_sampling: LightSamplingPDF) -> PDF {
        let emitter_id = self
            .emitters
            .iter()
            .position(|m| Arc::ptr_eq(light_sampling.mesh, m))
            .unwrap();
        // FIXME: As for now, we only support surface light, the PDF measure is always SA
        PDF::SolidAngle(
            light_sampling.mesh.direct_pdf(light_sampling) * self.emitters_cdf.pdf(emitter_id),
        )
    }
    pub fn sample_light(
        &self,
        p: &Point3<f32>,
        r_sel: f32,
        r: f32,
        uv: Point2<f32>,
    ) -> LightSampling {
        // Select the point on the light
        let (pdf_sel, emitter) = self.random_select_emitter(r_sel);
        let sampled_pos = emitter.sample(r, uv);

        // Compute the distance
        let mut d: Vector3<f32> = sampled_pos.p - p;
        let dist = d.magnitude();
        d /= dist;

        // Compute the geometry
        let cos_light = sampled_pos.n.dot(-d).max(0.0);
        let pdf = if cos_light == 0.0 {
            PDF::SolidAngle(0.0)
        } else {
            PDF::SolidAngle((pdf_sel * sampled_pos.pdf * dist * dist) / cos_light)
        };
        let emission = if pdf.is_zero() {
            Color::zero()
        } else {
            emitter.emission / pdf.value()
        };
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
}
