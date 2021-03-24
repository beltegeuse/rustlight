use crate::bsdfs;
use crate::bsdfs::*;
use crate::camera::Camera;
use crate::emitter::*;
use crate::geometry;
use crate::scene::*;
use crate::structure::*;
use cgmath::*;
#[cfg(feature = "pbrt")]
use pbrt_rs;
use serde_json;
use std::collections::HashMap;
use std::error::Error;
use std::io::Read;
use std::rc::Rc;

pub trait SceneLoader {
    fn load(&self, filename: &str) -> Result<Scene, Box<dyn Error>>;
}
pub struct SceneLoaderManager {
    loader: HashMap<String, Rc<dyn SceneLoader>>,
}
impl SceneLoaderManager {
    pub fn register(&mut self, name: &str, loader: Rc<dyn SceneLoader>) {
        self.loader.insert(name.to_string(), loader);
    }
    pub fn load(&self, filename: String) -> Result<Scene, Box<dyn Error>> {
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
    fn load(&self, filename: &str) -> Result<Scene, Box<dyn Error>> {
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

        // Read the object
        let obj_path_str: String = v["meshes"].as_str().unwrap().to_string();
        let obj_path = wk.join(obj_path_str);
        let mut meshes = geometry::load_obj(obj_path.as_path())?;

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
            meshes,
            nb_samples: 1,
            nb_threads: None,
            output_img_path: "out.pfm".to_string(),
            emitter_environment: None,
            volume: None,
        })
    }
}

#[cfg(feature = "pbrt")]
fn convert_spectrum_to_color(
    s: &pbrt_rs::parser::Spectrum,
    scale: Option<&pbrt_rs::parser::RGB>,
) -> Color {
    let scale = scale.map_or(Color::one(), |s| Color {
        r: s.r,
        g: s.g,
        b: s.b,
    });
    match s {
        pbrt_rs::parser::Spectrum::RGB(rgb) => Color::new(rgb.r, rgb.g, rgb.b) * scale,
        _ => panic!("convert_spectrum_to_color failed: {:?}", s),
    }
}

#[cfg(feature = "pbrt")]
pub struct PBRTSceneLoader {}
#[cfg(feature = "pbrt")]
impl SceneLoader for PBRTSceneLoader {
    fn load(&self, filename: &str) -> Result<Scene, Box<dyn Error>> {
        let mut scene_info = pbrt_rs::Scene::default();
        let mut state = pbrt_rs::State::default();
        pbrt_rs::read_pbrt_file(filename, &mut scene_info, &mut state);
        let wk = std::path::Path::new(filename).parent().unwrap();

        // Then do some transformation
        // if it is necessary
        for s in &mut scene_info.shapes {
            match &mut s.data {
                pbrt_rs::Shape::Ply { filename, .. } => {
                    s.data = pbrt_rs::ply::read_ply(std::path::Path::new(filename)).to_trimesh();
                }
                _ => (),
            }
        }

        // Load the data
        let materials = scene_info.materials;
        let textures = scene_info.textures;
        let meshes = scene_info
            .shapes
            .into_iter()
            .map(|m| match m.data {
                pbrt_rs::Shape::TriMesh {
                    uv,
                    normals,
                    points,
                    indices,
                } => {
                    let mat = m.matrix;
                    let reverse_orientation = m.reverse_orientation;
                    let normals = match normals {
                        Some(ref v) => Some(
                            v.iter()
                                .map(|n| {
                                    mat.transform_vector(if reverse_orientation {
                                        -n.clone()
                                    } else {
                                        n.clone()
                                    })
                                })
                                .collect(),
                        ),
                        None => None,
                    };
                    let points = points
                        .into_iter()
                        .map(|n| mat.transform_point(n.clone()).to_vec())
                        .collect();

                    let bsdf = if let Some(ref name) = m.material_name {
                        if let Some(bsdf_name) = materials.get(name) {
                            bsdfs::bsdf_pbrt(bsdf_name, &textures)
                        } else {
                            Box::new(bsdfs::diffuse::BSDFDiffuse {
                                diffuse: bsdfs::BSDFColor::UniformColor(Color::value(0.5)),
                            })
                        }
                    } else {
                        Box::new(bsdfs::diffuse::BSDFDiffuse {
                            diffuse: bsdfs::BSDFColor::UniformColor(Color::value(0.5)),
                        })
                    };
                    let mut mesh =
                        geometry::Mesh::new("noname".to_string(), points, indices, normals, uv);

                    mesh.bsdf = bsdf;
                    if m.emission.is_some() {
                        mesh.emission = convert_spectrum_to_color(m.emission.as_ref().unwrap(), None);
                    }
                    Some(mesh)
                }
                _ => {
                    warn!("All mesh should be converted to trimesh: {:?}", m.data);
                    None
                }
            })
            .filter(|x| x.is_some())
            .map(|m| m.unwrap())
            .collect::<Vec<_>>();


        // Check if there is other emitter type
        let emitter_environment = None;

        let camera = {
            if let Some(camera) = scene_info.cameras.get(0) {
                match camera {
                    pbrt_rs::Camera::Perspective {
                        world_to_camera,
                        fov,
                    } => {
                        let mat = world_to_camera.inverse_transform().unwrap();
                        info!("camera matrix: {:?}", mat);
                        Camera::new(scene_info.image_size, *fov, mat)
                    }
                }
            } else {
                panic!("The camera is not set!");
            }
        };
        camera.print_info();

        info!("image size: {:?}", scene_info.image_size);
        Ok(Scene {
            camera,
            meshes,
            nb_samples: 1,
            nb_threads: None,
            output_img_path: "out.pfm".to_string(),
            emitter_environment,
            volume: None,
        })
    }
}
