use crate::bsdfs;
use crate::bsdfs::*;
use crate::camera::{Camera, Fov};
use crate::emitter::*;
use crate::geometry;
use crate::scene::*;
use crate::structure::*;
use cgmath::Transform;
use cgmath::*;
#[cfg(feature = "mitsuba")]
use mitsuba_rs;
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
        #[cfg(feature = "pbrt")]
        loaders.register("pbrt", Rc::new(PBRTSceneLoader {}));
        #[cfg(feature = "mitsuba")]
        loaders.register("xml", Rc::new(MTSSceneLoader {}));
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
                Camera::new(img, Fov::Y(fov), matrix, false)
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
pub struct PBRTSceneLoader {}
#[cfg(feature = "pbrt")]
impl SceneLoader for PBRTSceneLoader {
    fn load(&self, filename: &str) -> Result<Scene, Box<dyn Error>> {
        let mut scene_info = pbrt_rs::Scene::default();
        let mut state = pbrt_rs::State::default();
        let working_dir = std::path::Path::new(filename).parent().unwrap();
        pbrt_rs::read_pbrt_file(filename, Some(&working_dir), &mut scene_info, &mut state);

        // Load the data
        let mut meshes: Vec<geometry::Mesh> = scene_info
            .shapes
            .iter()
            .map(|m| match m.data {
                pbrt_rs::Shape::TriMesh(ref data) => {
                    let mat = m.matrix;
                    let uv = data.uv.clone();
                    let normals = match data.normals {
                        Some(ref v) => {
                            Some(v.iter().map(|n| mat.transform_vector(n.clone())).collect())
                        }
                        None => None,
                    };
                    let points = data
                        .points
                        .iter()
                        .map(|n| mat.transform_point(n.clone()).to_vec())
                        .collect();
                    let indices = data.indices.clone();

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
                    let mut mesh =
                        geometry::Mesh::new("noname".to_string(), points, indices, normals, uv);
                    mesh.bsdf = bsdf;
                    mesh
                }
            })
            .collect();

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
                                emitter_environment = Some(EnvironmentLight {
                                    luminance: Color::new(rgb.r, rgb.g, rgb.b),
                                    world_radius: 1.0, // TODO: Add the correct radius
                                    world_position: Point3::new(0.0, 0.0, 0.0), // TODO:
                                });
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
                        Camera::new(scene_info.image_size, Fov::Y(cam.fov), mat, false)
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

#[cfg(feature = "mitsuba")]
pub struct MTSSceneLoader {}
#[cfg(feature = "mitsuba")]
impl SceneLoader for MTSSceneLoader {
    fn load(&self, filename: &str) -> Result<Scene, Box<dyn Error>> {
        // Load the scene
        let mut mts = mitsuba_rs::parse(filename);
        let wk = std::path::Path::new(filename).parent().unwrap();

        // Load camera
        let camera = {
            assert_eq!(mts.sensors.len(), 1);
            let mts_sensor = mts.sensors.pop().unwrap();
            let img_size = Vector2::new(mts_sensor.film.width, mts_sensor.film.height);
            let mat = mts_sensor.to_world.as_matrix();
            let mat = mat.inverse_transform().unwrap();
            // TODO: Revisit the sensor definition
            let fov = match &mts_sensor.fov_axis[..] {
                "x" => Fov::Y(mts_sensor.fov),
                "y" => Fov::X(mts_sensor.fov),
                _ => panic!("Unsupport Fov axis definition: {}", mts_sensor.fov_axis),
            };
            Camera::new(img_size, fov, mat, true)
        };

        // Helper to transform the mesh
        let apply_transform = |m: &mut geometry::Mesh, trans: Option<mitsuba_rs::Transform>| {
            // No transformation
            if trans.is_none() {
                return;
            }

            let mat = trans.unwrap().as_matrix();

            if let Some(v) = m.normals.as_mut() {
                for n in v.iter_mut() {
                    *n = mat.transform_vector(*n);
                }
            }
            for p in &mut m.vertices {
                *p = mat.transform_vector(*p);
            }
        };

        // Load meshes
        let mts_shapes = mts
            .shapes_id
            .into_iter()
            .map(|(_, v)| v)
            .chain(mts.shapes_unamed.into_iter());
        let meshes = mts_shapes
            .map(|s| {
                match s {
                    mitsuba_rs::Shape::Obj {
                        filename,
                        option,
                        flip_tex_coords,
                        ..
                    } => {
                        let obj_path = wk.join(std::path::Path::new(&filename));

                        // Load the geometry
                        let mut meshes = geometry::load_obj(&obj_path).unwrap();

                        // apply BSDF
                        for m in &mut meshes {
                            m.bsdf = match &option.bsdf {
                                Some(bsdf) => crate::bsdfs::bsdf_mts(bsdf, wk),
                                None => Box::new(crate::bsdfs::diffuse::BSDFDiffuse {
                                    diffuse: crate::bsdfs::BSDFColor::UniformColor(Color::value(
                                        0.8,
                                    )),
                                }),
                            };
                        }

                        // Check if a emitter is attached
                        if let Some(v) = option.emitter {
                            for m in &mut meshes {
                                let rgb = v.radiance.clone().as_rgb();
                                m.emission = Color {
                                    r: rgb.r,
                                    g: rgb.g,
                                    b: rgb.b,
                                };
                            }
                        }

                        if flip_tex_coords {
                            for m in &mut meshes {
                                if let Some(uv) = m.uv.as_mut() {
                                    for e in uv {
                                        e.x = 1.0 - e.x;
                                        e.y = 1.0 - e.y;
                                    }
                                }
                            }
                        }

                        // Apply transform
                        for m in &mut meshes {
                            apply_transform(m, option.to_world.clone());
                        }

                        meshes
                    }
                    mitsuba_rs::Shape::Serialized(shape) => {
                        let mut mesh_mts = mitsuba_rs::serialized::read_serialized(&shape, &wk);

                        // Build CDF
                        let mut dist_const =
                            crate::math::Distribution1DConstruct::new(mesh_mts.indices.len());
                        for id in &mesh_mts.indices {
                            let v0 = mesh_mts.vertices[id.x];
                            let v1 = mesh_mts.vertices[id.y];
                            let v2 = mesh_mts.vertices[id.z];

                            let area = (v1 - v0).cross(v2 - v0).magnitude() * 0.5;
                            dist_const.add(area);
                        }

                        // Normalize normals
                        // Indeed, sometimes the normal are not properly normalized
                        if let Some(ref mut ns) = mesh_mts.normals.as_mut() {
                            for n in ns.iter_mut() {
                                let l = n.dot(*n);
                                if l == 0.0 {
                                    warn!("Wrong normal! {:?}", n);
                                // TODO: Need to do something...
                                } else if l != 1.0 {
                                    *n /= l.sqrt();
                                }
                            }
                        }

                        let mut meshes = vec![geometry::Mesh {
                            name: mesh_mts.name, // Does this is the name to use?
                            vertices: mesh_mts.vertices,
                            indices: mesh_mts.indices,
                            normals: mesh_mts.normals,
                            uv: mesh_mts.texcoords,
                            bsdf: match &shape.option.bsdf {
                                Some(bsdf) => crate::bsdfs::bsdf_mts(bsdf, wk),
                                None => Box::new(crate::bsdfs::diffuse::BSDFDiffuse {
                                    diffuse: crate::bsdfs::BSDFColor::UniformColor(Color::value(
                                        0.8,
                                    )),
                                }),
                            },
                            emission: match &shape.option.emitter {
                                Some(emitter) => {
                                    let rgb = emitter.radiance.clone().as_rgb();
                                    Color {
                                        r: rgb.r,
                                        g: rgb.g,
                                        b: rgb.b,
                                    }
                                }
                                None => Color::zero(),
                            },
                            cdf: dist_const.normalize(),
                        }];

                        // Apply transform
                        for m in &mut meshes {
                            apply_transform(m, shape.option.to_world.clone());
                        }

                        meshes
                    }
                    _ => {
                        warn!("Ignoring shape {:?}", s);
                        vec![]
                    }
                }
            })
            .flatten()
            .collect();

        // Other
        let emitter_environment = None;

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
