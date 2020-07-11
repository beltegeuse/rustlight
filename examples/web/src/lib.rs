mod utils;
use cgmath::prelude::*;
use cgmath::Point3;
use rustlight::bsdfs::*;
use rustlight::camera::*;
use rustlight::emitter::*;
use rustlight::geometry::*;
use rustlight::integrators::IntegratorMC;
use rustlight::structure::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen::Clamped;
use web_sys::{CanvasRenderingContext2d, ImageData};

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet(name: &str) {
    alert(&format!("Hello, rustlight! (from {})", name));
}

#[wasm_bindgen]
pub struct PixelValue {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

#[wasm_bindgen]
pub struct Scene {
    scene: rustlight::scene::Scene,
    sampler: rustlight::samplers::independent::IndependentSampler,
    int: rustlight::integrators::explicit::path::IntegratorPathTracing,
    pub width: u32,
    pub height: u32,
    img: Vec<Color>,
    nb_samples: Vec<usize>,
}

#[wasm_bindgen]
impl Scene {
    #[wasm_bindgen(constructor)]
    pub fn new(object: &JsValue) -> Result<Scene, JsValue> {
        console_error_panic_hook::set_once();
        let scene_desc = object.as_string().unwrap();

        // Read from PBRT
        let (scene, (width, height)) = {
            let mut scene_info = pbrt_rs::Scene::default();
            let mut state = pbrt_rs::State::default();
            pbrt_rs::read_pbrt(&scene_desc, None, &mut scene_info, &mut state);

            // Load the data
            let mut meshes: Vec<rustlight::geometry::Mesh> = scene_info
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
                                rustlight::bsdfs::bsdf_pbrt(bsdf_name, &scene_info)
                            } else {
                                Box::new(rustlight::bsdfs::diffuse::BSDFDiffuse {
                                    diffuse: rustlight::bsdfs::BSDFColor::UniformColor(
                                        Color::value(0.8),
                                    ),
                                })
                            }
                        } else {
                            Box::new(rustlight::bsdfs::diffuse::BSDFDiffuse {
                                diffuse: rustlight::bsdfs::BSDFColor::UniformColor(Color::value(
                                    0.8,
                                )),
                            })
                        };
                        let mut mesh =
                            Mesh::new("noname".to_string(), points, indices, normals, uv);
                        mesh.bsdf = bsdf;
                        mesh
                    }
                })
                .collect();

            // Assign materials and emissions
            for (i, shape) in scene_info.shapes.iter().enumerate() {
                match shape.emission {
                    Some(pbrt_rs::Param::RGB(ref rgb)) => {
                        // info!("assign emission: RGB({},{},{})", rgb.r, rgb.g, rgb.b);
                        meshes[i].emission = Color::new(rgb.r, rgb.g, rgb.b)
                    }
                    None => {}
                    _ => {} // _ => warn!("unsupported emission profile: {:?}", shape.emission),
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
                                    // warn!("Unsupported luminance field: {:?}", infinite.luminance);
                                }
                            }
                        }
                        _ => {
                            // warn!("Igoring light type: {:?}", l);
                        }
                    }
                }
            };

            let camera = {
                if let Some(camera) = scene_info.cameras.get(0) {
                    match camera {
                        pbrt_rs::Camera::Perspective(ref cam) => {
                            let mat = cam.world_to_camera.inverse_transform().unwrap();
                            // info!("camera matrix: {:?}", mat);
                            Camera::new(scene_info.image_size, cam.fov, mat)
                        }
                    }
                } else {
                    panic!("The camera is not set!");
                }
            };
            camera.print_info();

            // info!("image size: {:?}", scene_info.image_size);
            (
                rustlight::scene::Scene {
                    camera,
                    meshes,
                    nb_samples: 1,
                    nb_threads: None,
                    output_img_path: "out.pfm".to_string(),
                    emitter_environment,
                    volume: None,
                },
                (scene_info.image_size.x, scene_info.image_size.y),
            )
        };

        Ok(Scene {
            scene,
            sampler: rustlight::samplers::independent::IndependentSampler::from_seed(0),
            int: rustlight::integrators::explicit::path::IntegratorPathTracing {
                min_depth: None,
                max_depth: None,
                strategy:
                    rustlight::integrators::explicit::path::IntegratorPathTracingStrategies::All,
                single_scattering: false,
            },
            width,
            height,
            img: vec![Color::default(); (width * height) as usize],
            nb_samples: vec![0; (width * height) as usize],
        })
    }

    pub fn render_block(&mut self, x: u32, y: u32, size_x: u32, size_y: u32) {
        // Create accel structure
        let accel = rustlight::accel::NaiveAcceleration::new(&self.scene);
        let light_sampling = self.scene.emitters_sampler();

        for x in x..(x + size_x).min(self.width) {
            for y in y..(y + size_y).min(self.height) {
                let c = self.int.compute_pixel(
                    (x, y),
                    &accel,
                    &self.scene,
                    &mut self.sampler,
                    &light_sampling,
                );
                let pixel_index = (y * self.width + x) as usize;
                self.img[pixel_index] += c;
                self.nb_samples[pixel_index] += 1;
            }
        }
    }

    pub fn get_img(&self, ctx: &CanvasRenderingContext2d) -> Result<(), JsValue> {
        let mut data: Vec<u8> = vec![0; (self.width * self.height) as usize * 4];
        for i in 0..(self.width * self.height) as usize {
            if self.nb_samples[i] != 0 {
                let inv_nb_samples = 1.0 / self.nb_samples[i] as f32;
                data[i * 4 + 0] = ((self.img[i].r * inv_nb_samples).powf(1.0 / 2.2).min(1.0) * 255.0) as u8;
                data[i * 4 + 1] = ((self.img[i].g * inv_nb_samples).powf(1.0 / 2.2).min(1.0) * 255.0) as u8;
                data[i * 4 + 2] = ((self.img[i].b * inv_nb_samples).powf(1.0 / 2.2).min(1.0) * 255.0) as u8;
                data[i * 4 + 3] = 255;
            } else {
                data[i * 4 + 0] = 0;
                data[i * 4 + 1] = 0;
                data[i * 4 + 2] = 0;
                data[i * 4 + 3] = 255;
            }
        }
        let data = ImageData::new_with_u8_clamped_array_and_sh(
            Clamped(&mut data),
            self.width,
            self.height,
        )?;
        ctx.put_image_data(&data, 0.0, 0.0)
    }
}

#[wasm_bindgen(start)]
pub fn main() -> Result<(), JsValue> {
    // Use `web_sys`'s global `window` function to get a handle on the global
    // window object.
    // let window = web_sys::window().expect("no global `window` exists");
    // let document = window.document().expect("should have a document on window");
    // let body = document.body().expect("document should have a body");

    // // Manufacture the element we're gonna append
    // let val = document.create_element("p")?;
    // val.set_inner_html("Hello from Rust!");

    // body.append_child(&val)?;

    Ok(())
}
