use crate::bsdfs;
use crate::bsdfs::*;
use crate::camera::Camera;
use crate::geometry;
use crate::math::{Distribution1D, Distribution1DConstruct};
use crate::structure::*;
use crate::emitter::*;
use cgmath::*;
use embree_rs;
#[cfg(feature = "pbrt")]
use pbrt_rs;
use serde_json;
use std;
use std::error::Error;

/// Scene representation
pub struct Scene {
    /// Main camera
    pub camera: Camera,
    pub nb_samples: usize,
    pub nb_threads: Option<usize>,
    pub output_img_path: String,
    // Geometry information
    pub meshes: Vec<geometry::Mesh>,
    pub emitters: Vec<Emitter>,
    emitters_cdf: Distribution1D,
    embree_scene: embree_rs::Scene,
    emitter_environment: Option<usize>, // ID from the emitters
}

impl Scene {
    pub fn nb_samples(&self) -> usize {
        self.nb_samples
    }

    #[cfg(not(feature = "pbrt"))]
    pub fn pbrt(
        filename: &str,
        nb_samples: usize,
        nb_threads: Option<usize>,
        output_img_path: String,
    ) -> Result<Scene, Box<Error>> {
        panic!("Rustlight wasn't compiled with pbrt file support.")
    }
    #[cfg(feature = "pbrt")]
    pub fn pbrt(
        filename: &str,
        nb_samples: usize,
        nb_threads: Option<usize>,
        output_img_path: String,
    ) -> Result<Scene, Box<Error>> {
        let mut scene_info = pbrt_rs::Scene::default();
        let mut state = pbrt_rs::State::default();
        let working_dir = std::path::Path::new(filename).parent().unwrap();
        pbrt_rs::read_pbrt_file(filename, &working_dir, &mut scene_info, &mut state);

        // Allocate embree
        let device = embree_rs::Device::debug();
        let mut scene_embree = embree_rs::SceneConstruct::new(&device);

        // Load the data
        let mut meshes: Vec<Box<geometry::Mesh>> = scene_info
            .shapes
            .iter()
            .map(|m| match m.data {
                pbrt_rs::Shape::TriMesh(ref data) => {
                    let mat = m.matrix;
                    let uv = if let Some(uv) = data.uv.clone() {
                        uv
                    } else {
                        vec![]
                    };
                    let normals = match data.normals {
                        Some(ref v) => v.iter().map(|n| mat.transform_vector(n.clone())).collect(),
                        None => Vec::new(),
                    };
                    let points = data
                        .points
                        .iter()
                        .map(|n| mat.transform_point(n.clone()))
                        .collect();
                    let trimesh = scene_embree.add_triangle_mesh(
                        &device,
                        points,
                        normals,
                        uv,
                        data.indices.clone(),
                    );

                    let bsdf = if let Some(ref name) = m.material_name {
                        if let Some(bsdf_name) = scene_info.materials.get(name) {
                            bsdfs::bsdf_pbrt(bsdf_name, &scene_info)
                        } else {
                            Box::new(bsdfs::diffuse::BSDFDiffuse {
                                diffuse: bsdfs::BSDFColor::UniformColor(Color::value(0.8)),
                            })
                        }
                    } else {
                        Box::new(bsdfs::diffuse::BSDFDiffuse {
                            diffuse: bsdfs::BSDFColor::UniformColor(Color::value(0.8)),
                        })
                    };

                    // FIXME FIXME
                    Box::new(geometry::Mesh::new("noname".to_string(), trimesh, bsdf))
                }
            })
            .collect();
        info!("Build the acceleration structure");
        let scene_embree = scene_embree.commit()?;

        // Assign materials and emissions
        for (i, shape) in scene_info.shapes.iter().enumerate() {
            match shape.emission {
                Some(pbrt_rs::Param::RGB(ref rgb)) => {
                    info!("assign emission: RGB({},{},{})", rgb.r, rgb.g, rgb.b);
                    meshes[i].emission = Color::new(rgb.r, rgb.g, rgb.b)
                }
                None => {}
                _ => warn!("unsupported emission profile: {:?}", shape.emission),
            }
        }

        let meshes: Vec<Arc<geometry::Mesh>> = meshes.into_iter().map(|e| Arc::from(e)).collect();

        // Update the list of lights
        let mut emitters = meshes
            .iter()
            .filter(|m| !m.emission.is_zero())
            .map(|m| Emitter::Mesh(m.clone()))
            .collect::<Vec<_>>();

        // Check if there is other emitter type 
        let mut emitter_environment = None;
        {
            let mut have_env = false;
            for l in scene_info.lights {
                match l {
                    pbrt_rs::Light::Infinite(ref infinite) => {
                        match infinite.luminance {
                            pbrt_rs::Param::RGB(ref rgb) => {
                                if have_env {
                                    panic!("Multiple env map is NOT supported");
                                }
                                emitters.push(Emitter::Environment(EnvironmentLight {
                                    luminance: Color::new(rgb.r, rgb.g, rgb.b),
                                    world_radius: 1.0, // TODO: Add the correct radius
                                }));
                                emitter_environment = Some(emitters.len() - 1);
                                have_env = true;
                            }
                            _ => {
                                warn!("Unsupported luminance field: {:?}", infinite.luminance);
                            }
                        }
                    }
                    _ => {
                        warn!("Igoring light type: {:?}", l);
                    }
                }
            }
        };

        let emitters_cdf = {
            let mut cdf_construct = Distribution1DConstruct::new(emitters.len());
            emitters
                .iter()
                .map(|e| match e {
                    Emitter::Environment(ref env) => env.flux(), 
                    Emitter::Mesh(ref e) => e.flux(),
                })
                .for_each(|f| cdf_construct.add(f.channel_max()));
            cdf_construct.normalize()
        };
        if emitters_cdf.normalization == 0.0 {
            warn!("no light attached to the scene. Only AO will works");
        }

        let camera = {
            if let Some(camera) = scene_info.cameras.get(0) {
                match camera {
                    pbrt_rs::Camera::Perspective(ref cam) => {
                        let mat = cam.world_to_camera.inverse_transform().unwrap();
                        info!("camera matrix: {:?}", mat);
                        Camera::new(scene_info.image_size, cam.fov, mat)
                    }
                }
            } else {
                panic!("The camera is not set!");
            }
        };


        info!("image size: {:?}", scene_info.image_size);
        Ok(Scene {
            camera,
            embree_scene: scene_embree,
            meshes,
            emitters,
            emitters_cdf,
            nb_samples,
            nb_threads,
            output_img_path,
            emitter_environment,
        })
    }

    /// Take a json formatted string and an working directory
    /// and build the scene representation.
    pub fn json(
        data: &str,
        wk: &std::path::Path,
        nb_samples: usize,
        nb_threads: Option<usize>,
        output_img_path: String,
    ) -> Result<Scene, Box<Error>> {
        // Read json string
        let v: serde_json::Value = serde_json::from_str(data)?;

        // Allocate embree
        let device = embree_rs::Device::debug();
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
        info!("Emitters:");
        if let Some(emitters_json) = v.get("emitters") {
            for e in emitters_json.as_array().unwrap() {
                let name: String = e["mesh"].as_str().unwrap().to_string();
                let emission: Color = serde_json::from_value(e["emission"].clone())?;
                info!(" - emission: {}", name);
                // Get the set of matched meshes
                let mut matched_meshes = meshes
                    .iter_mut()
                    .filter(|m| m.name == name)
                    .collect::<Vec<_>>();
                match matched_meshes.len() {
                    0 => panic!("Not found {} in the obj list", name),
                    1 => {
                        matched_meshes[0].emission = emission;
                        info!("   * flux: {:?}", matched_meshes[0].flux());
                    }
                    _ => panic!("Several {} in the obj list", name),
                };
            }
        }
        // - BSDF
        info!("BSDFS:");
        if let Some(bsdfs_json) = v.get("bsdfs") {
            for b in bsdfs_json.as_array().unwrap() {
                let name: String = serde_json::from_value(b["mesh"].clone())?;
                info!(" - replace bsdf: {}", name);
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

        info!("Build vectors and Discrete CDF");
        // Transform the scene mesh from Box to Arc
        let meshes: Vec<Arc<geometry::Mesh>> = meshes.into_iter().map(|e| Arc::from(e)).collect();

        // Update the list of lights & construct the CDF
        let emitters = meshes
            .iter()
            .filter(|m| !m.emission.is_zero())
            .map(|m| Emitter::Mesh(m.clone()))
            .collect::<Vec<_>>();
        let emitters_cdf = {
            let mut cdf_construct = Distribution1DConstruct::new(emitters.len());
            emitters
                .iter()
                .map(|e| match e {
                    Emitter::Environment(ref env) => env.flux(), 
                    Emitter::Mesh(ref e) => e.flux(),
                })
                .for_each(|f| cdf_construct.add(f.channel_max()));
            cdf_construct.normalize()
        };
        info!(
            "CDF lights: {:?} norm: {:?}",
            emitters_cdf.cdf, emitters_cdf.normalization
        );

        // Read the camera config
        let camera = {
            if let Some(camera_json) = v.get("camera") {
                let fov: f32 = serde_json::from_value(camera_json["fov"].clone())?;
                let img: Vector2<u32> = serde_json::from_value(camera_json["img"].clone())?;
                let m: Vec<f32> = serde_json::from_value(camera_json["matrix"].clone())?;

                //let matrix = Matrix4::new(
                //    m[0], m[4], m[8], m[12], m[1], m[5], m[9], m[13], m[2], m[6], m[10], m[14],
                //    m[3], m[7], m[11], m[15],
                //);
                let matrix = Matrix4::new(
                    m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11],
                    m[12], m[13], m[14], m[15],
                );

                info!("m: {:?}", matrix);
                Camera::new(img, fov, matrix)
            } else {
                panic!("The camera is not set!");
            }
        };
        camera.print_info();

        // Define a default scene
        Ok(Scene {
            camera,
            embree_scene: scene_embree,
            meshes,
            emitters,
            emitters_cdf,
            nb_samples,
            nb_threads,
            output_img_path,
            emitter_environment: None,
        })
    }

    /// Intersect and compute intersection information
    pub fn trace(&self, ray: &Ray) -> Option<Intersection> {
        match self.embree_scene.intersect(ray.to_embree()) {
            None => None,
            Some(its) => {
                let geom_id = its.geom_id as usize;
                Some(Intersection::new(&its, -ray.d, &self.meshes[geom_id]))
            }
        }
    }
    pub fn visible(&self, p0: &Point3<f32>, p1: &Point3<f32>) -> bool {
        let d = p1 - p0;
        !self
            .embree_scene
            .occluded(embree_rs::Ray::new(*p0, d).near(0.00001).far(0.9999))
    }

    pub fn direct_pdf(&self, light_sampling: &LightSamplingPDF) -> PDF {
        let emitter_id = self
            .emitters
            .iter()
            .position({|m| 
                match m {
                    Emitter::Mesh(ref m) => Arc::ptr_eq(light_sampling.mesh, m),
                    _ => false
                }
            })
            .unwrap();
        // FIXME: As for now, we only support surface light, the PDF measure is always SA
        PDF::SolidAngle(
            light_sampling.mesh.direct_pdf(&light_sampling) * self.emitters_cdf.pdf(emitter_id),
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
        let emitter = match emitter {
            Emitter::Mesh(ref emitter) => {
                emitter
            }
            _ => {
                panic!("Sample position is not supported")
            }
        };
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
    pub fn random_select_emitter(&self, v: f32) -> (f32, &Emitter) {
        let id_light = self.emitters_cdf.sample(v);
        (self.emitters_cdf.pdf(id_light), &self.emitters[id_light])
    }

    pub fn random_sample_emitter_position(
        &self,
        v1: f32,
        v2: f32,
        uv: Point2<f32>,
    ) -> (&Arc<geometry::Mesh>, PDF, geometry::SampledPosition) {
        let (pdf_sel, emitter) = self.random_select_emitter(v1);
        match emitter {
            Emitter::Mesh(ref emitter) => {
                let sampled_pos = emitter.sample(v2, uv);
                (emitter, PDF::Area(pdf_sel * sampled_pos.pdf), sampled_pos)
            }
            _ => {
                panic!("random sample position is not supported")
            }
        }
        
    }

    pub fn enviroment_luminance(&self, d: Vector3<f32>) -> Color {
        match self.emitter_environment {
            None => Color::zero(),
            Some(id) => {
                if let Emitter::Environment(ref env) = self.emitters[id] {
                    env.emitted_luminance(d)
                } else {
                    panic!("Wrong ID for the env emitter");
                }
            }
        }
    }
}
