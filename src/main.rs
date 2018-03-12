extern crate image;
extern crate rayon;
extern crate rustlight;
extern crate cgmath;
extern crate byteorder;
extern crate clap;

use image::*;
use std::time::Instant;
use std::io::prelude::*;
use cgmath::Point2;
use byteorder::{WriteBytesExt, LittleEndian};
use clap::{Arg, App, SubCommand};

fn main() {
    // Read input args
    let matches = App::new("rustlight")
        .version("0.0.2")
        .author("Adrien Gruson <adrien.gruson@gmail.com>")
        .about("A Rusty Light Transport simulation program")
        .arg(Arg::with_name("scene")
            .required(true)
            .takes_value(true)
            .index(1)
            .help("JSON file description"))
        .arg(Arg::with_name("output")
            .takes_value(true)
            .short("o")
            .default_value("test.pfm")
            .help("output image file"))
        .arg(Arg::with_name("integrator")
            .takes_value(true)
            .default_value("path")
            .help("integration technique"))
        .arg(Arg::with_name("nbsamples")
            .short("n")
            .takes_value(true)
            .default_value("128")
            .help("integration technique"))
        .get_matches();

    /////////////// Check output extension
    let imgout_path_str = matches.value_of("output").unwrap();
    let output_ext = match std::path::Path::new(imgout_path_str).extension() {
        None => panic!("No file extension provided"),
        Some(x) => std::ffi::OsStr::to_str(x).unwrap(),
    };

    //////////////// Load the rendering configuration
    // Create the integrator
    let nb_samples = match matches.value_of("nbsamples").unwrap().parse::<i32>() {
        Err(e) => panic!("Error while parsing the number of samples"),
        Ok(x) => x,
    };
    let int: Box<rustlight::integrator::Integrator + Sync + Send> = match matches.value_of("integrator").unwrap() {
        "path" => Box::new(rustlight::integrator::IntergratorPath {
            max_depth: 10
        }),
        "ao" => Box::new(rustlight::integrator::IntergratorAO { max_distance: None, }),
        "direct" => Box::new(rustlight::integrator::IntergratorDirect {
            nb_bsdf_samples: 1,
            nb_light_samples: 1
        }),
        _ => panic!("unknown integrator"),
    };

    //////////////// Load the scene
    let scene_path = std::path::Path::new(matches.value_of("scene").unwrap());
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
    match output_ext {
        "pfm" => {
            let mut file = std::fs::File::create(std::path::Path::new(imgout_path_str)).unwrap();
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
        },
        "png" => {
            // The image that we will render
            let mut image_ldr = DynamicImage::new_rgb8(scene.camera.size().x,
                                                       scene.camera.size().y);
            for x in 0..img.size.x {
                for y in 0..img.size.y {
                    image_ldr.put_pixel(x, y,
                                        img.get(Point2::new(img.size.x - x - 1, y)).to_rgba())
                }
            }
            let ref mut fout = std::fs::File::create(imgout_path_str).unwrap();
            image_ldr.save(fout, image::PNG).expect("failed to write img into file");
        }
        _ => panic!("Unknow output file extension"),
    }
}
