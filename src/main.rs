// For computing the time
use std::time::Instant;
use std::io::prelude::*;

// Vector math library
extern crate cgmath;

use cgmath::*;

// For saving image (ldr)
extern crate image;

use image::*;

extern crate tobj;

extern crate rayon;

#[macro_use]
extern crate serde_json;

use serde_json::Value;

extern crate embree;

extern crate rustlight;

use rustlight::rustlight::*;
use rustlight::rustlight::structure::*;

fn load_obj(scene: &mut embree::rtcore::Scene, file_name: &std::path::Path) {
    println!("Try to load {:?}", file_name);
    match tobj::load_obj(file_name) {
        Ok((models, _)) => {
            for m in models {
                println!("Loading model {}", m.name);
                let mesh = m.mesh;
                println!("{} has {} triangles", m.name, mesh.indices.len() / 3);
                let verts = mesh.positions.chunks(3).map(|i| Vector4::new(i[0], i[1], i[2], 1.0)).collect();
                scene.new_triangle_mesh(embree::rtcore::GeometryFlags::Static,
                                        verts,
                                        mesh.indices);
                // TODO: Load also normals and uv (chunks(3) and chunks(2))
                // TODO: Only load them by checking if the vec is empty
            }
        }
        Err(e) => {
            println!("Failed to load {:?} due to {:?}", file_name, e);
        }
    };
}

#[allow(dead_code)]
fn make_cube(scene: &mut embree::rtcore::Scene) {
    let verts = vec![Vector4::new(-1.0, -1.0, -1.0, 0.0),
                     Vector4::new(-1.0, -1.0, 1.0, 0.0),
                     Vector4::new(-1.0, 1.0, -1.0, 0.0),
                     Vector4::new(-1.0, 1.0, 1.0, 0.0),
                     Vector4::new(1.0, -1.0, -1.0, 0.0),
                     Vector4::new(1.0, -1.0, 1.0, 0.0),
                     Vector4::new(1.0, 1.0, -1.0, 0.0),
                     Vector4::new(1.0, 1.0, 1.0, 0.0)];

    let vidxs = vec![0, 2, 1,
                     1, 2, 3,
                     4, 5, 6,
                     5, 7, 6,
                     0, 1, 4,
                     1, 5, 4,
                     2, 6, 3,
                     3, 6, 7,
                     0, 4, 2,
                     2, 4, 6,
                     1, 3, 5,
                     3, 7, 5];

    scene.new_triangle_mesh(embree::rtcore::GeometryFlags::Static,
                            verts,
                            vidxs);
}

#[allow(dead_code)]
fn make_ground_plane(scene: &mut embree::rtcore::Scene) {
    let vlist: Vec<Vector4<f32>> = vec![Vector4 {
        x: 1.0,
        y: 1.0,
        z: 1.0,
        w: 1.0,
    },
    Vector4 {
        x: 2.0,
        y: 1.0,
        z: 1.0,
        w: 0.0,
    },
    Vector4 {
        x: 2.0,
        y: 2.0,
        z: 1.0,
        w: 1.0,
    },
    Vector4 {
        x: 1.0,
        y: 2.0,
        z: 1.0,
        w: 1.0,
    }];

    // Indices
    let vidxs: Vec<u32> = vec![0, 1, 3, 1, 2, 3];

    scene.new_triangle_mesh(embree::rtcore::GeometryFlags::Static,
                            vlist,
                            vidxs);
}

fn main() {

    // Read the scene file
    let mut fscene = std::fs::File::open("scene.json").expect("scene file not found");
    let mut scene_str = String::new();
    fscene.read_to_string(&mut scene_str).expect("impossible to read the file");
    let v: Value = serde_json::from_str(&scene_str).expect("impossible to parse in JSON");

    let json_camera = json!(v.get("camera").unwrap());
    let camera_param: rustlight::rustlight::camera::CameraParam = serde_json::from_value(json_camera).unwrap();

    // Read the geometries
    let mut device = embree::rtcore::Device::new();
    let mut scene_embree = device.new_scene(embree::rtcore::STATIC,
                                            embree::rtcore::INTERSECT1 | embree::rtcore::INTERPOLATE);

    //make_cube(&mut scene_embree);
    //make_ground_plane(&mut scene_embree);
    let obj_path = std::path::Path::new("./data/dragon.obj");
    load_obj(&mut scene_embree, obj_path);
    println!("Build the acceleration structure");
    scene_embree.commit(); // Build

    // Define a default scene
    let scene = scene::Scene {
        camera: rustlight::rustlight::camera::Camera::new(camera_param),
        embree: &scene_embree,
        bsdf: vec![Color::new(1.0, 0.0, 0.0), Color::new(0.0, 0.0, 1.0), Color::new(0.0, 1.0, 0.0)],
        nb_samples: 32,
    };

    // To time the rendering time
    let start = Instant::now();
    // Generate the thread pool
    let pool = rayon::ThreadPool::new(rayon::Configuration::new()).unwrap();
    // Render the image
    let img: DynamicImage = pool.install(|| scene::render(&scene));

    assert_eq!(scene.camera.size().x, img.width());
    assert_eq!(scene.camera.size().y, img.height());
    // Compute the rendering time
    let elapsed = start.elapsed();
    println!("Elapsed: {} ms",
             (elapsed.as_secs() * 1_000) + (elapsed.subsec_nanos() / 1_000_000) as u64);
    // Save the image
    let ref mut fout = std::fs::File::create("test.png").unwrap();
    img.save(fout, image::PNG).expect("failed to write img into file");
}
