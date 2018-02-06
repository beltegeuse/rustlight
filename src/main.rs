// For computing the time
use std::time::Instant;

// Vector math library
extern crate cgmath;

use cgmath::*;

// For saving image (ldr)
extern crate image;

use image::*;

extern crate rayon;

extern crate rustlight;

use rustlight::rustlight::*;
use rustlight::rustlight::sampler::Sampler;

fn main() {
    // Create some geometries (randomly)
    let mut sampler = sampler::IndepSampler::new();
    let mut spheres = Vec::new();
    for _ in 0..100000 {
        let s1 = geometry::Sphere {
            pos: Point3 {
                x: sampler.next() * 10.0 - 5.0,
                y: sampler.next() * 10.0 - 5.0,
                z: -5.0 - sampler.next(),
            },
            radius: 0.5,
            color: structure::Color {
                r: sampler.next(),
                g: sampler.next(),
                b: sampler.next(),
            },
        };
        spheres.push(s1);
    }


    // Define a default scene
    let scene = scene::Scene {
        camera: rustlight::rustlight::camera::Camera::look_at(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, -5.0),
            Vector3::new(0.0, 1.0, 0.0),
            90.0, Vector2 { x: 800, y: 600 }),
        spheres: spheres,
    };

    // To time the rendering time
    let start = Instant::now();
    // Generate the thread pool
    let pool = rayon::ThreadPool::new(rayon::Configuration::new()).unwrap();
    // Render the image
    let img: DynamicImage = pool.install(|| scene::render(&scene));
    assert_eq!(scene.camera.img.x, img.width());
    assert_eq!(scene.camera.img.y, img.height());
    // Compute the rendering time
    let elapsed = start.elapsed();
    println!("Elapsed: {} ms",
             (elapsed.as_secs() * 1_000) + (elapsed.subsec_nanos() / 1_000_000) as u64);
    // Save the image
    let ref mut fout = std::fs::File::create("test.png").unwrap();
    img.save(fout, image::PNG).expect("failed to write img into file");
}
