extern crate image;
extern crate rayon;
extern crate rustlight;

use image::*;
use std::time::Instant;
use std::io::prelude::*;

fn main() {
    // TODO: Need to be an argument of the application
    let scene_path = std::path::Path::new("./data/cbox.json");

    // Load the scene
    // - read the file
    let mut fscene = std::fs::File::open(scene_path).expect("scene file not found");
    let mut data = String::new();
    fscene.read_to_string(&mut data).expect("impossible to read the file");
    // - build the scene
    let wk = scene_path.parent().expect("impossible to extract parent directory for OBJ loading");
    let scene = rustlight::scene::Scene::new(data, wk).expect("error when loading the scene");

    println!("Rendering...");
    let start = Instant::now();
    let pool = rayon::ThreadPool::new(rayon::Configuration::new()).unwrap();
    let img: DynamicImage = pool.install(|| scene.render());
    let elapsed = start.elapsed();
    println!("Elapsed: {} ms",
             (elapsed.as_secs() * 1_000) + (elapsed.subsec_nanos() / 1_000_000) as u64);

    // Save the image
    let ref mut fout = std::fs::File::create("test.png").unwrap();
    img.save(fout, image::PNG).expect("failed to write img into file");
}
