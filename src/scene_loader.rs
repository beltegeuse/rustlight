use crate::bsdfs;
use crate::bsdfs::*;
use crate::camera::Camera;
use crate::emitter::*;
use crate::geometry;
use crate::math::Distribution1DConstruct;
use crate::scene::Scene;
use crate::structure::*;
use cgmath::*;
#[cfg(feature = "pbrt")]
use pbrt_rs;
use serde_json;
use std::collections::HashMap;
use std::error::Error;
use std::io::Read;
use std::rc::Rc;
use std::sync::Arc;

pub trait SceneLoader {
    fn load(&self, filename: &str) -> Result<Scene, Box<Error>>;
}
pub struct SceneLoaderManager {
    loader: HashMap<String, Rc<dyn SceneLoader>>,
}
impl SceneLoaderManager {
    pub fn register(&mut self, name: &str, loader: Rc<dyn SceneLoader>) {
        self.loader.insert(name.to_string(), loader);
    }
    pub fn load(&self, filename: String) -> Result<Scene, Box<Error>> {
        let filename_ext = match std::path::Path::new(&filename).extension() {
            None => panic!("No file extension provided"),
            Some(x) => std::ffi::OsStr::to_str(x).expect("Issue to unpack the file"),
        };
        if let Some(loader) = self.loader.get(filename_ext) {
            loader.load(&filename)
        } else {
            panic!(
                "Impossible to found scene loader for {} extension",
                filename_ext
            )
        }
    }
}
impl Default for SceneLoaderManager {
    fn default() -> Self {
        let mut loaders = SceneLoaderManager {
            loader: HashMap::default(),
        };
        loaders.register("json", Rc::new(JSONSceneLoader {}));
        if cfg!(feature = "pbrt") {
            loaders.register("pbrt", Rc::new(PBRTSceneLoader {}));
        }
        loaders
    }
}

pub struct JSONSceneLoader {}
impl SceneLoader for JSONSceneLoader {
    fn load(&self, filename: &str) -> Result<Scene, Box<Error>> {
        // Reading the scene
        let scene_path = std::path::Path::new(filename);
        let mut fscene = std::fs::File::open(scene_path).expect("scene file not found");
        let mut data = String::new();
        fscene
            .read_to_string(&mut data)
            .expect("impossible to read the file");
        let wk = scene_path
            .parent()
            .expect("impossible to extract parent directory for OBJ loading");

        // Read json string
        let v: serde_json::Value = serde_json::from_str(&data)?;

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
        let meshes = meshes.into_iter().map(|e| Arc::from(e)).collect::<Vec<_>>();

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
        let mut scene = Scene {
            camera,
            embree_scene: scene_embree,
            meshes,
            nb_samples: 1,
            nb_threads: None,
            output_img_path: "out.pfm".to_string(),
            emitter_environment: None,
        };
        Ok(scene)
    }
}

#[cfg(feature = "pbrt")]
pub struct PBRTSceneLoader {}
#[cfg(feature = "pbrt")]
impl SceneLoader for PBRTSceneLoader {
    fn load(&self, filename: &str) -> Result<Scene, Box<Error>> {
        let mut scene_info = pbrt_rs::Scene::default();
        let mut state = pbrt_rs::State::default();
        let working_dir = std::path::Path::new(filename).parent().unwrap();
        pbrt_rs::read_pbrt_file(filename, &working_dir, &mut scene_info, &mut state);

        // Allocate embree
        let device = embree_rs::Device::debug();
        let mut scene_embree = embree_rs::SceneConstruct::new(&device);

        // Load the data
        let mut meshes: Vec<geometry::Mesh> = scene_info
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
                    geometry::Mesh::new("noname".to_string(), trimesh, bsdf)
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
                                emitter_environment = Some(Arc::new(EnvironmentLight {
                                    luminance: Color::new(rgb.r, rgb.g, rgb.b),
                                    world_radius: 1.0, // TODO: Add the correct radius
                                    world_position: Point3::new(0.0, 0.0, 0.0), // TODO:
                                }));
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
            nb_samples: 1,
            nb_threads: None,
            output_img_path: "out.pfm".to_string(),
            emitter_environment,
        })
    }
}
