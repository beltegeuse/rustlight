use crate::camera::{Camera, Fov};
use crate::emitter::*;
use crate::geometry;
use crate::scene::*;
use crate::structure::*;
use crate::{bsdfs, geometry::Mesh};
use cgmath::Transform;
use cgmath::*;
#[cfg(feature = "mitsuba")]
use mitsuba_rs;
#[cfg(feature = "pbrt")]
use pbrt_rs;
use std::collections::HashMap;
use std::error::Error;
use std::rc::Rc;
use std::sync::Arc;

pub trait SceneLoader {
    fn load(&self, filename: &str, use_shading_normal: bool) -> Result<Scene, Box<dyn Error>>;
}
pub struct SceneLoaderManager {
    loader: HashMap<String, Rc<dyn SceneLoader>>,
}
impl SceneLoaderManager {
    pub fn register(&mut self, name: &str, loader: Rc<dyn SceneLoader>) {
        self.loader.insert(name.to_string(), loader);
    }
    pub fn load(
        &self,
        filename: String,
        use_shading_normal: bool,
    ) -> Result<Scene, Box<dyn Error>> {
        let filename_ext = match std::path::Path::new(&filename).extension() {
            None => panic!("No file extension provided"),
            Some(x) => std::ffi::OsStr::to_str(x).expect("Issue to unpack the file"),
        };
        if let Some(loader) = self.loader.get(filename_ext) {
            loader.load(&filename, use_shading_normal)
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
        #[cfg(feature = "pbrt")]
        loaders.register("pbrt", Rc::new(PBRTSceneLoader {}));
        #[cfg(feature = "mitsuba")]
        loaders.register("xml", Rc::new(MTSSceneLoader {}));
        loaders
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
impl PBRTSceneLoader {
    fn transform_mesh(
        mut m: pbrt_rs::ShapeInfo,
        materials: &HashMap<String, pbrt_rs::BSDF>,
        textures: &HashMap<String, pbrt_rs::Texture>,
        matrix: Matrix4<f32>,
        use_shading_normal: bool,
    ) -> Option<Mesh> {
        match m.data {
            pbrt_rs::Shape::Ply { filename, .. } => {
                m.data =
                    pbrt_rs::ply::read_ply(std::path::Path::new(&filename), false).to_trimesh();
                PBRTSceneLoader::transform_mesh(m, materials, textures, matrix, use_shading_normal)
            }
            pbrt_rs::Shape::TriMesh {
                uv,
                normals,
                points,
                indices,
            } => {
                let mat = matrix * m.matrix;
                let reverse_orientation = m.reverse_orientation;
                let normals = if use_shading_normal {
                    match normals {
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
                    }
                } else {
                    None
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
                            diffuse: bsdfs::BSDFColor::Constant(Color::value(0.5)),
                        })
                    }
                } else {
                    Box::new(bsdfs::diffuse::BSDFDiffuse {
                        diffuse: bsdfs::BSDFColor::Constant(Color::value(0.5)),
                    })
                };
                let mesh = geometry::Mesh::new("noname".to_string(), points, indices, normals, uv);

                if let Some(mut mesh) = mesh {
                    mesh.bsdf = bsdf;
                    if m.emission.is_some() {
                        mesh.emission = geometry::EmissionType::Color {
                            v: convert_spectrum_to_color(m.emission.as_ref().unwrap(), None),
                        };
                    }
                    Some(mesh)
                } else {
                    None
                }
            }
            _ => {
                warn!("All mesh should be converted to trimesh: {:?}", m.data);
                None
            }
        }
    }
}
#[cfg(feature = "pbrt")]
impl SceneLoader for PBRTSceneLoader {
    fn load(&self, filename: &str, use_shading_normal: bool) -> Result<Scene, Box<dyn Error>> {
        let mut scene_info = pbrt_rs::Scene::default();
        let mut state = pbrt_rs::State::default();
        pbrt_rs::read_pbrt_file(filename, &mut scene_info, &mut state);
        let wk = std::path::Path::new(filename).parent().unwrap();

        // Load the data
        let materials = scene_info.materials;
        let textures = scene_info.textures;
        let mut meshes = scene_info
            .shapes
            .into_iter()
            .map(|m| {
                PBRTSceneLoader::transform_mesh(
                    m,
                    &materials,
                    &textures,
                    Matrix4::identity(),
                    use_shading_normal,
                )
            })
            .filter(|x| x.is_some())
            .map(|m| m.unwrap())
            .collect::<Vec<_>>();

        for instance in scene_info.instances {
            let object = scene_info.objects.get(&instance.name).unwrap();
            meshes.extend(
                object
                    .shapes
                    .clone()
                    .into_iter()
                    .map(|m| {
                        PBRTSceneLoader::transform_mesh(
                            m,
                            &materials,
                            &textures,
                            instance.matrix,
                            use_shading_normal,
                        )
                    })
                    .filter(|x| x.is_some())
                    .map(|m| m.unwrap()),
            );
        }

        // Check if there is other emitter type
        let mut emitters: Vec<Box<dyn Emitter>> = Vec::new();
        let mut emitter_environment = None;
        {
            let mut have_env = false;
            for l in scene_info.lights {
                match l {
                    pbrt_rs::Light::Distant {
                        luminance,
                        from,
                        to,
                        scale,
                    } => {
                        let luminance = convert_spectrum_to_color(&luminance, Some(&scale));
                        let direction = (to - from).normalize();
                        emitters.push(Box::new(crate::emitter::DirectionalLight {
                            direction,
                            intensity: luminance,
                            bsphere: None,
                        }));
                    }
                    pbrt_rs::Light::Point {
                        intensity,
                        from,
                        scale,
                    } => {
                        let intensity = convert_spectrum_to_color(&intensity, Some(&scale));
                        dbg!(&from);
                        emitters.push(Box::new(crate::emitter::PointEmitter {
                            intensity,
                            position: from,
                        }));
                    }
                    pbrt_rs::Light::Infinite {
                        luminance, scale, ..
                    } => match &luminance {
                        pbrt_rs::parser::Spectrum::RGB(rgb) => {
                            if have_env {
                                panic!("Multiple env map is NOT supported");
                            }
                            emitter_environment = Some(Arc::new(EnvironmentLight {
                                luminance: EnvironmentLightColor::Constant(Color::new(
                                    rgb.r * scale.r,
                                    rgb.g * scale.g,
                                    rgb.b * scale.b,
                                )),
                                bsphere: None,
                            }));
                            have_env = true;
                        }
                        pbrt_rs::parser::Spectrum::Texture(name) => {
                            dbg!(name);
                            todo!()
                        }
                        pbrt_rs::parser::Spectrum::Mapname(name) => {
                            assert!(scale.r == 1.0);
                            assert!(scale.g == 1.0);
                            assert!(scale.b == 1.0);

                            let filename = wk.join(name);
                            let image = Bitmap::read(filename.to_str().unwrap());
                            emitter_environment = Some(Arc::new(EnvironmentLight {
                                luminance: EnvironmentLightColor::new_texture(image),
                                bsphere: None,
                            }));
                        }
                        _ => {
                            warn!("Unsupported luminance field: {:?}", luminance);
                        }
                    },
                }
            }
        };

        let camera = {
            if let Some(camera) = scene_info.cameras.get(0) {
                match camera {
                    pbrt_rs::Camera::Perspective {
                        world_to_camera,
                        fov,
                    } => {
                        let mat = world_to_camera.inverse_transform().unwrap();
                        info!("camera matrix: {:?}", mat);
                        info!("camera fov: {:?}", fov);
                        Camera::new(scene_info.image_size, Fov::Y(*fov), mat, false)
                    }
                }
            } else {
                panic!("The camera is not set!");
            }
        };
        camera.print_info();

        let meshes = meshes.into_iter().map(|v| Arc::new(v)).collect();

        info!("image size: {:?}", scene_info.image_size);
        Ok(Scene {
            camera,
            meshes,
            nb_samples: 1,
            nb_threads: None,
            output_img_path: "out.pfm".to_string(),
            emitter_environment,
            volume: None,
            emitters: Some(EmittersState::Unbuild(emitters)),
            bsphere: None,
        })
    }
}

#[cfg(feature = "mitsuba")]
pub struct MTSSceneLoader {}
#[cfg(feature = "mitsuba")]
impl SceneLoader for MTSSceneLoader {
    fn load(&self, filename: &str, use_shading_normal: bool) -> Result<Scene, Box<dyn Error>> {
        // Load the scene
        let mut mts = mitsuba_rs::parse(filename)?;
        let wk = std::path::Path::new(filename).parent().unwrap();

        // Load camera
        let camera = {
            assert_eq!(mts.sensors.len(), 1);
            let mts_sensor = mts.sensors.pop().unwrap();
            let img_size = Vector2::new(mts_sensor.film.width, mts_sensor.film.height);
            let mat = mts_sensor.to_world.as_matrix();
            let fov = match &mts_sensor.fov_axis[..] {
                "x" => Fov::X(mts_sensor.fov),
                "y" => Fov::Y(mts_sensor.fov),
                _ => panic!("Unsupport Fov axis definition: {}", mts_sensor.fov_axis),
            };
            Camera::new(img_size, fov, mat, true)
        };

        // Helper to transform the mesh
        let apply_transform = |m: &mut geometry::Mesh, trans: Option<mitsuba_rs::Transform>| {
            // No transformation
            if trans.is_none() {
                if let Some(v) = m.normals.as_mut() {
                    for n in v.iter_mut() {
                        let l = n.dot(*n);
                        if l == 0.0 {
                            warn!("Wrong normal! {:?}", n);
                        } else if l != 1.0 {
                            *n /= l.sqrt();
                        }
                    }
                }
                // Build CDF for be able to sample the mesh
                m.build_cdf();
                return;
            }

            // Apply transform
            let mat = trans.unwrap().as_matrix();
            if let Some(v) = m.normals.as_mut() {
                for n in v.iter_mut() {
                    *n = mat.transform_vector(*n);
                    let l = n.dot(*n);
                    if l == 0.0 {
                        warn!("Wrong normal! {:?}", n);
                    } else if l != 1.0 {
                        *n /= l.sqrt();
                    }
                }
            }
            for p in &mut m.vertices {
                *p = mat.transform_point(Point3::new(p.x, p.y, p.z)).to_vec();
            }

            // Build CDF for be able to sample the mesh
            m.build_cdf();
        };

        // Load meshes
        let mts_shapes = mts
            .shapes_id
            .into_iter()
            .map(|(_, v)| v)
            .chain(mts.shapes_unamed.into_iter());
        let meshes: Vec<geometry::Mesh> = mts_shapes
            .map(|s| {
                match s {
                    mitsuba_rs::Shape::Ply {
                        filename,
                        face_normal,
                        option,
                        ..
                    } => {
                        let mut ply_path = std::path::Path::new(&filename).to_owned();
                        if !ply_path.is_absolute() {
                            ply_path = wk.join(ply_path);
                        }

                        let mesh_mts = mitsuba_rs::ply::read_ply(&ply_path);
                        let vertices = mesh_mts.points.into_iter().map(|p| p.to_vec()).collect();
                        let mut meshes = vec![geometry::Mesh {
                            name: "".to_owned(), // Does this is the name to use?
                            vertices,
                            indices: mesh_mts.indices,
                            normals: if face_normal || !use_shading_normal {
                                None
                            } else {
                                mesh_mts.normals
                            },
                            uv: mesh_mts.uv,
                            bsdf: match &option.bsdf {
                                Some(bsdf) => crate::bsdfs::bsdf_mts(bsdf, wk),
                                None => Box::new(crate::bsdfs::diffuse::BSDFDiffuse {
                                    diffuse: crate::bsdfs::BSDFColor::Constant(Color::value(0.8)),
                                }),
                            },
                            emission: match &option.emitter {
                                Some(emitter) => {
                                    let rgb = emitter.radiance.clone().as_rgb().unwrap();
                                    geometry::EmissionType::Color {
                                        v: Color {
                                            r: rgb.r,
                                            g: rgb.g,
                                            b: rgb.b,
                                        },
                                    }
                                }
                                None => geometry::EmissionType::Zero,
                            },
                            cdf: None,
                        }];

                        // Apply transform
                        for m in &mut meshes {
                            apply_transform(m, option.to_world.clone());
                        }

                        meshes
                    }
                    mitsuba_rs::Shape::Obj {
                        filename,
                        option,
                        face_normal,
                        flip_tex_coords,
                        ..
                    } => {
                        let obj_path = wk.join(std::path::Path::new(&filename));

                        // Load the geometry
                        let mut meshes = geometry::load_obj(&obj_path).unwrap();

                        // apply BSDF & face_normal is needed
                        for m in &mut meshes {
                            if face_normal {
                                m.discard_normals();
                            }

                            m.bsdf = match &option.bsdf {
                                Some(bsdf) => crate::bsdfs::bsdf_mts(bsdf, wk),
                                None => Box::new(crate::bsdfs::diffuse::BSDFDiffuse {
                                    diffuse: crate::bsdfs::BSDFColor::Constant(Color::value(0.8)),
                                }),
                            };
                        }

                        // Check if a emitter is attached
                        if let Some(v) = option.emitter {
                            for m in &mut meshes {
                                let rgb = v.radiance.clone().as_rgb().unwrap();
                                m.emission = geometry::EmissionType::Color {
                                    v: Color {
                                        r: rgb.r,
                                        g: rgb.g,
                                        b: rgb.b,
                                    },
                                };
                            }
                        }

                        if flip_tex_coords {
                            for m in &mut meshes {
                                if let Some(uv) = m.uv.as_mut() {
                                    for e in uv {
                                        // Only v coordinate
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
                        let mesh_mts = mitsuba_rs::serialized::read_serialized(&shape, &wk);
                        let mut meshes = vec![geometry::Mesh {
                            name: mesh_mts.name, // Does this is the name to use?
                            vertices: mesh_mts.vertices,
                            indices: mesh_mts.indices,
                            normals: if shape.face_normal {
                                None
                            } else {
                                mesh_mts.normals
                            },
                            uv: mesh_mts.texcoords,
                            bsdf: match &shape.option.bsdf {
                                Some(bsdf) => crate::bsdfs::bsdf_mts(bsdf, wk),
                                None => Box::new(crate::bsdfs::diffuse::BSDFDiffuse {
                                    diffuse: crate::bsdfs::BSDFColor::Constant(Color::value(0.8)),
                                }),
                            },
                            emission: match &shape.option.emitter {
                                Some(emitter) => {
                                    let rgb = emitter.radiance.clone().as_rgb().unwrap();
                                    geometry::EmissionType::Color {
                                        v: Color {
                                            r: rgb.r,
                                            g: rgb.g,
                                            b: rgb.b,
                                        },
                                    }
                                }
                                None => geometry::EmissionType::Zero,
                            },
                            cdf: None,
                        }];

                        // Apply transform
                        for m in &mut meshes {
                            apply_transform(m, shape.option.to_world.clone());
                        }

                        meshes
                    }
                    mitsuba_rs::Shape::Rectangle { option } => {
                        // Temporary support of rectangular shape:
                        //  - Convert to triangle soup
                        // This prevent for now uses of special sampling techniques.
                        let vertices = vec![
                            Vector3::new(-1.0, -1.0, 0.0),
                            Vector3::new(1.0, -1.0, 0.0),
                            Vector3::new(1.0, 1.0, 0.0),
                            Vector3::new(-1.0, 1.0, 0.0),
                        ];
                        let uv = Some(vec![
                            Vector2::new(0.0, 0.0),
                            Vector2::new(1.0, 0.0),
                            Vector2::new(1.0, 1.0),
                            Vector2::new(0.0, 1.0),
                        ]);
                        let normals = Some(vec![
                            Vector3::new(0.0, 0.0, 1.0),
                            Vector3::new(0.0, 0.0, 1.0),
                            Vector3::new(0.0, 0.0, 1.0),
                            Vector3::new(0.0, 0.0, 1.0),
                        ]);
                        let indices = vec![Vector3::new(0, 1, 2), Vector3::new(2, 3, 0)];
                        let mut meshes = vec![geometry::Mesh {
                            name: "rectangle".to_string(), // Does this is the name to use?
                            vertices,
                            indices,
                            normals,
                            uv,
                            bsdf: match &option.bsdf {
                                Some(bsdf) => crate::bsdfs::bsdf_mts(bsdf, wk),
                                None => Box::new(crate::bsdfs::diffuse::BSDFDiffuse {
                                    diffuse: crate::bsdfs::BSDFColor::Constant(Color::value(0.8)),
                                }),
                            },
                            emission: match &option.emitter {
                                Some(emitter) => {
                                    let rgb = emitter.radiance.clone().as_rgb().unwrap();
                                    geometry::EmissionType::Color {
                                        v: Color {
                                            r: rgb.r,
                                            g: rgb.g,
                                            b: rgb.b,
                                        },
                                    }
                                }
                                None => geometry::EmissionType::Zero,
                            },
                            cdf: None,
                        }];

                        // Apply transform
                        for m in &mut meshes {
                            apply_transform(m, option.to_world.clone());
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
        let mut emitters: Vec<Box<dyn Emitter>> = Vec::new();
        {
            for x in mts.emitters {
                match x {
                    mitsuba_rs::Emitter::Point {
                        position,
                        intensity,
                        to_world,
                        ..
                    } => {
                        let position = to_world.as_matrix().transform_point(position);
                        let rgb = intensity.clone().as_rgb().unwrap();
                        emitters.push(Box::new(crate::emitter::PointEmitter {
                            intensity: Color {
                                r: rgb.r,
                                g: rgb.g,
                                b: rgb.b,
                            },
                            position,
                        }))
                    }
                    mitsuba_rs::Emitter::PointNormal {
                        position,
                        intensity,
                        normal,
                        to_world,
                        ..
                    } => {
                        let trans = to_world.as_matrix();
                        let position = trans.transform_point(position);

                        // Get the normal and normalize it
                        let normal = trans.transform_vector(normal);
                        let normal_l = normal.magnitude();
                        dbg!(&normal);
                        assert!(normal_l != 0.0);
                        let normal = normal / normal_l;

                        let rgb = intensity.clone().as_rgb().unwrap();
                        emitters.push(Box::new(crate::emitter::PointNormalEmitter {
                            intensity: Color {
                                r: rgb.r,
                                g: rgb.g,
                                b: rgb.b,
                            },
                            position,
                            normal,
                        }))
                    }
                    _ => {
                        warn!("Ignoring emitter");
                    }
                }
            }
        };

        let meshes = meshes.into_iter().map(|v| Arc::new(v)).collect();
        let volume = if mts.medium.is_empty() {
            None
        } else {
            let (medium_name, medium) = mts.medium.iter().nth(0).unwrap();
            info!("Add {} as main medium", medium_name);
            match medium {
                mitsuba_rs::Medium::Homogenous {
                    sigma_s,
                    sigma_a,
                    scale,
                    phase,
                } => {
                    let sigma_a = sigma_a.clone().as_rgb().unwrap();
                    let sigma_a = Color {
                        r: sigma_a.r * scale,
                        g: sigma_a.g * scale,
                        b: sigma_a.b * scale,
                    };
                    let sigma_s = sigma_s.clone().as_rgb().unwrap();
                    let sigma_s = Color {
                        r: sigma_s.r * scale,
                        g: sigma_s.g * scale,
                        b: sigma_s.b * scale,
                    };
                    info!(
                        "Create homogenous PM with sigma_a = {:?} and sigma_s = {:?}",
                        sigma_a, sigma_s
                    );

                    let phase = match phase {
                        mitsuba_rs::PhaseFunction::Isotropic => {
                            crate::volume::PhaseFunction::Isotropic()
                        }
                        mitsuba_rs::PhaseFunction::HG { g } => {
                            crate::volume::PhaseFunction::HenyeyGreenstein(*g)
                        }
                    };

                    let sigma_t = sigma_a + sigma_s;
                    Some(crate::volume::HomogenousVolume {
                        sigma_a,
                        sigma_s,
                        sigma_t,
                        phase,
                    })
                }
            }
        };

        Ok(Scene {
            camera,
            meshes,
            nb_samples: 1,
            nb_threads: None,
            output_img_path: "out.pfm".to_string(),
            emitter_environment,
            volume,
            emitters: Some(EmittersState::Unbuild(emitters)),
            bsphere: None,
        })
    }
}
