extern crate image;
extern crate rayon;
extern crate rustlight;
extern crate cgmath;
extern crate byteorder;

use image::*;
use std::time::Instant;
use std::io::prelude::*;
use cgmath::Point2;
use byteorder::{WriteBytesExt, LittleEndian};

fn main() {
    // TODO: Need to be an argument of the application
    let scene_path = std::path::Path::new("./data/cbox.json");

    // Create the integrator
    let nb_samples = 128;
    let int = Box::new(rustlight::integrator::IntergratorPath {
        max_depth : 10
    });
//    let int = Box::new(rustlight::integrator::IntergratorAO { max_distance: None, } );
//    let int = Box::new(rustlight::integrator::IntergratorDirect {
//        nb_bsdf_samples : 1,
//        nb_light_samples : 1
//    });

    // Load the scene
    // - read the file
    let mut fscene = std::fs::File::open(scene_path).expect("scene file not found");
    let mut data = String::new();
    fscene.read_to_string(&mut data).expect("impossible to read the file");
    // - build the scene
    let wk = scene_path.parent().expect("impossible to extract parent directory for OBJ loading");
    let scene = rustlight::scene::Scene::new(&data, wk, int).expect("error when loading the scene");

    println!("Rendering...");
    let start = Instant::now();
    let pool = rayon::ThreadPool::new(rayon::Configuration::new()).unwrap();
    let img = pool.install(|| scene.render(nb_samples));
    let elapsed = start.elapsed();
    println!("Elapsed: {} ms",
             (elapsed.as_secs() * 1_000) + (elapsed.subsec_nanos() / 1_000_000) as u64);

    // Save the image (HDR and LDF)
    // -- LDR
    {
        // The image that we will render
        let mut image_ldr = DynamicImage::new_rgb8(scene.camera.size().x,
                                                   scene.camera.size().y);
        for x in 0..img.size.x {
            for y in 0..img.size.y {
                image_ldr.put_pixel(x, y,
                                    img.get(Point2::new(img.size.x - x - 1, y)).to_rgba())
            }
        }
        let ref mut fout = std::fs::File::create("test.png").unwrap();
        image_ldr.save(fout, image::PNG).expect("failed to write img into file");
    }
    // - HDR
    {
        let mut file = std::fs::File::create(std::path::Path::new("test.pfm")).unwrap();
        let header = format!("PF\n{} {}\n-1.0\n",
                             img.size.y, img.size.x);
        file.write(header.as_bytes()).unwrap();
        for y in 0..img.size.y {
            for x in 0..img.size.x {
                let p = img.get(Point2::new(img.size.x - x - 1, img.size.y - y - 1));
                file.write_f32::<LittleEndian>(p.r).unwrap();
                file.write_f32::<LittleEndian>(p.g).unwrap();
                file.write_f32::<LittleEndian>(p.b).unwrap();
            }
        }
    }
}
