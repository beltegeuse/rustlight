// For computing the time
use std::time::Instant;
use std::io::prelude::*;

// Vector math library
extern crate cgmath;
use cgmath::*;

// For saving image (ldr)
extern crate image;
use image::*;

extern crate rayon;

#[macro_use]
extern crate serde_json;
use serde_json::Value;

extern crate embree;

extern crate rustlight;
use rustlight::rustlight::*;
use rustlight::rustlight::structure::*;

fn make_cube<'a,'b>(scene: &'b mut embree::rtcore::Scene<'a>) -> embree::rtcore::GeometryHandle<'a> {
    let verts = vec![ Vector4::new(-1.0, -1.0, -1.0, 0.0),
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

    let hm: embree::rtcore::GeometryHandle<'a> = scene.new_triangle_mesh(embree::rtcore::GeometryFlags::Static,
                            verts,
                            vidxs);
    hm
}

fn make_ground_plane<'a,'b>(scene: &'b mut embree::rtcore::Scene<'a>) -> embree::rtcore::GeometryHandle<'a> {
    let verts = vec![
        Vector4::new(-10.0, -2.0, -10.0, 0.0),
        Vector4::new(-10.0, -2.0, 10.0, 0.0),
        Vector4::new(10.0, -2.0, 10.0, 0.0),
        Vector4::new(10.0, -2.0, -10.0, 0.0)
    ];

    let vidxs = vec![0, 1, 2, 1, 2, 3];

    let hm: embree::rtcore::GeometryHandle<'a> = scene.new_triangle_mesh(embree::rtcore::GeometryFlags::Static,
                                                                         verts,
                                                                         vidxs);
    hm
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
    let mut scene_embree = device.new_scene(embree::rtcore::STATIC, embree::rtcore::INTERSECT1);

    make_cube(&mut scene_embree);
    make_ground_plane(& mut scene_embree);

    scene_embree.commit(); // Build

    // Read materials
    let face_colors = vec![Color { r: 1.0, g: 0.0, b: 0.0 },
                           Color { r: 1.0, g: 0.0, b: 0.0 },
                           Color { r: 0.0, g: 1.0, b: 0.0 },
                           Color { r: 0.0, g: 1.0, b: 0.0 },
                           Color { r: 0.5, g: 0.5, b: 0.5 },
                           Color { r: 0.5, g: 0.5, b: 0.5 },
                           Color { r: 1.0, g: 1.0, b: 1.0 },
                           Color { r: 1.0, g: 1.0, b: 1.0 },
                           Color { r: 0.0, g: 0.0, b: 1.0 },
                           Color { r: 0.0, g: 0.0, b: 1.0 },
                           Color { r: 1.0, g: 1.0, b: 0.0 },
                           Color { r: 1.0, g: 1.0, b: 0.0 },
    ];

    // Define a default scene
    let scene = scene::Scene {
        camera: rustlight::rustlight::camera::Camera::new(camera_param),
        embree: &scene_embree,
        bsdf: face_colors,
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
