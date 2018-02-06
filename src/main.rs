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
use rustlight::rustlight::sampler::Sampler;
use rustlight::rustlight::structure::*;

fn make_cube<'a>(scene: &'a embree::Scene) -> embree::TriangleMesh<'a> {
    let mut mesh = embree::TriangleMesh::unanimated(scene, embree::GeometryFlags::STATIC,
                                                    12, 8);
    {
        let mut verts = mesh.vertex_buffer.map();
        let mut tris = mesh.index_buffer.map();

        verts[0] = Vector4::new(-1.0, -1.0, -1.0, 0.0);
        verts[1] = Vector4::new(-1.0, -1.0, 1.0, 0.0);
        verts[2] = Vector4::new(-1.0, 1.0, -1.0, 0.0);
        verts[3] = Vector4::new(-1.0, 1.0, 1.0, 0.0);
        verts[4] = Vector4::new(1.0, -1.0, -1.0, 0.0);
        verts[5] = Vector4::new(1.0, -1.0, 1.0, 0.0);
        verts[6] = Vector4::new(1.0, 1.0, -1.0, 0.0);
        verts[7] = Vector4::new(1.0, 1.0, 1.0, 0.0);

        // left side
        tris[0] = Vector3::new(0, 2, 1);
        tris[1] = Vector3::new(1, 2, 3);

        // right side
        tris[2] = Vector3::new(4, 5, 6);
        tris[3] = Vector3::new(5, 7, 6);

        // bottom side
        tris[4] = Vector3::new(0, 1, 4);
        tris[5] = Vector3::new(1, 5, 4);

        // top side
        tris[6] = Vector3::new(2, 6, 3);
        tris[7] = Vector3::new(3, 6, 7);

        // front side
        tris[8] = Vector3::new(0, 4, 2);
        tris[9] = Vector3::new(2, 4, 6);

        // back side
        tris[10] = Vector3::new(1, 3, 5);
        tris[11] = Vector3::new(3, 7, 5);
    }
    mesh
}

fn make_ground_plane<'a>(scene: &'a embree::Scene) -> embree::QuadMesh<'a> {
    let mut mesh = embree::QuadMesh::unanimated(scene, embree::GeometryFlags::STATIC,
                                                1, 4);
    {
        let mut verts = mesh.vertex_buffer.map();
        let mut quads = mesh.index_buffer.map();
        verts[0] = Vector4::new(-10.0, -2.0, -10.0, 0.0);
        verts[1] = Vector4::new(-10.0, -2.0, 10.0, 0.0);
        verts[2] = Vector4::new(10.0, -2.0, 10.0, 0.0);
        verts[3] = Vector4::new(10.0, -2.0, -10.0, 0.0);

        quads[0] = Vector4::<i32>::new(0, 1, 2, 3);
    }
    mesh
}

fn main() {

    // Read the scene file
    let mut fscene = std::fs::File::open("scene.json").expect("scene file not found");
    let mut scene_str = String::new();
    fscene.read_to_string(&mut scene_str).unwrap();
    let v: Value = serde_json::from_str(&scene_str).unwrap();

    let json_camera = json!(v.get("camera").unwrap());
    let camera_param: rustlight::rustlight::camera::CameraParam = serde_json::from_value(json_camera).unwrap();

    // Read the geometries
    let device = embree::Device::new();
    let sceneEmbree = embree::Scene::new(&device, embree::SceneFlags::SCENE_STATIC,
                                         embree::AlgorithmFlags::INTERSECT1);
    let cube = make_cube(&sceneEmbree);
    let ground = make_ground_plane(&sceneEmbree);
    sceneEmbree.commit(); // Build

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
        embree: &sceneEmbree,
        bsdf: face_colors,
    };

    // To time the rendering time
    let start = Instant::now();
    // Generate the thread pool
    let pool = rayon::ThreadPool::new(rayon::Configuration::new()).unwrap();
    // Render the image
    //let img: DynamicImage = pool.install(|| scene::render(&scene));
    let img: DynamicImage = scene::render(&scene);
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
