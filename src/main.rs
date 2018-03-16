extern crate image;
extern crate rayon;
extern crate rustlight;
extern crate cgmath;
extern crate byteorder;
#[macro_use]
extern crate clap;

use image::*;
use std::time::Instant;
use std::io::prelude::*;
use cgmath::Point2;
use byteorder::{WriteBytesExt, LittleEndian};
use clap::{Arg, App, SubCommand};
use rustlight::structure::Color;
use rustlight::scene::Bitmap;
use rustlight::integrator::ColorGradient;

fn classical_mc_integration(scene: &rustlight::scene::Scene,
                            nb_samples: u32,
                            int: Box<rustlight::integrator::Integrator<Color> + Sync + Send>) -> Bitmap<Color> {
    ////////////// Do the rendering
    println!("Rendering...");
    let start = Instant::now();
    let pool = rayon::ThreadPoolBuilder::new().build().unwrap();
    let img = pool.install(|| scene.render(int, nb_samples));
    let elapsed = start.elapsed();
    println!("Elapsed: {} ms",
             (elapsed.as_secs() * 1_000) + (elapsed.subsec_nanos() / 1_000_000) as u64);

    return img;
}

fn gradient_domain_integration(scene: &rustlight::scene::Scene,
                               nb_samples: u32,
                               int: Box<rustlight::integrator::Integrator<ColorGradient> + Sync + Send>) -> Bitmap<Color> {

    println!("Rendering...");
    let start = Instant::now();
    let pool = rayon::ThreadPoolBuilder::new().build().unwrap();
    let img_grad = pool.install(|| scene.render(int, nb_samples));
    let elapsed = start.elapsed();
    println!("Elapsed: {} ms",
             (elapsed.as_secs() * 1_000) + (elapsed.subsec_nanos() / 1_000_000) as u64);

    // Do the reconstruction
    let mut image = Bitmap::new(Point2::new(0, 0), scene.camera.size().clone());
    for x in 0..img_grad.size.x {
        for y in 0..img_grad.size.y {
            let c_p = Point2::new(img_grad.pos.x + x, img_grad.pos.y + y);
            image.accum(c_p, &img_grad.get(Point2::new(x, y)).main);
        }
    }
    image
}

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
            .help("output image file"))
        .arg(Arg::with_name("nbsamples")
            .short("n")
            .takes_value(true)
            .help("integration technique"))
        .subcommand(SubCommand::with_name("path")
            .about("path tracing")
            .arg(Arg::with_name("max").takes_value(true).short("m").default_value("inf")))
        .subcommand(SubCommand::with_name("gd-path")
            .about("gradient-domain path tracing")
            .arg(Arg::with_name("max").takes_value(true).short("m").default_value("inf")))
        .subcommand(SubCommand::with_name("ao")
            .about("ambiant occlusion")
            .arg(Arg::with_name("distance").takes_value(true).short("d").default_value("inf")))
        .subcommand(SubCommand::with_name("direct")
            .about("direct lighting")
            .arg(Arg::with_name("bsdf").takes_value(true).short("b").default_value("1"))
            .arg(Arg::with_name("light").takes_value(true).short("l").default_value("1")))
        .get_matches();

    /////////////// Check output extension
    let imgout_path_str = matches.value_of("output").unwrap_or("test.pfm");
    let output_ext = match std::path::Path::new(imgout_path_str).extension() {
        None => panic!("No file extension provided"),
        Some(x) => std::ffi::OsStr::to_str(x).expect("Issue to unpack the file"),
    };

    //////////////// Load the rendering configuration
    let nb_samples = value_t_or_exit!(matches.value_of("nbsamples"), u32);

    //////////////// Load the scene
    let scene_path_str = matches.value_of("scene").expect("no scene parameter provided");
    let scene_path = std::path::Path::new(scene_path_str);
    // - read the file
    let mut fscene = std::fs::File::open(scene_path).expect("scene file not found");
    let mut data = String::new();
    fscene.read_to_string(&mut data).expect("impossible to read the file");
    // - build the scene
    let wk = scene_path.parent().expect("impossible to extract parent directory for OBJ loading");
    let scene = rustlight::scene::Scene::new(&data, wk).expect("error when loading the scene");

    let img = match matches.subcommand() {
        ("path", Some(m)) => {
            let max_str = m.value_of("max").unwrap();
            let max_depth: Option<u32> = match max_str {
                "inf" => None,
                _ => Some(max_str.parse::<u32>().expect("wrong distance"))
            };

            classical_mc_integration(&scene, nb_samples,
                                     Box::new(rustlight::integrator::IntegratorPath {
                max_depth,
            }))
        },
        ("gd-path", Some(m)) => {
            let max_str = m.value_of("max").unwrap();
            let max_depth: Option<u32> = match max_str {
                "inf" => None,
                _ => Some(max_str.parse::<u32>().expect("wrong distance"))
            };

            gradient_domain_integration(&scene, nb_samples,
                                     Box::new(rustlight::integrator::IntegratorPath {
                                         max_depth,
                                     }))
        },
        ("ao", Some(m)) => {
            let dist_str = m.value_of("distance").unwrap();
            let dist: Option<f32> = match dist_str {
                "inf" => None,
                _ => Some(dist_str.parse::<f32>().expect("wrong distance"))
            };
            classical_mc_integration( &scene, nb_samples,
                                     Box::new(rustlight::integrator::IntergratorAO {
                max_distance: dist,
            }))
        }
        ("direct", Some(m)) => classical_mc_integration(&scene,
                                                        nb_samples,
                                                        Box::new(rustlight::integrator::IntergratorDirect {
                                                            nb_bsdf_samples: value_t_or_exit!(m.value_of("bsdf"), u32),
                                                            nb_light_samples: value_t_or_exit!(m.value_of("light"), u32),
                                                        })),
        _ => panic!("unknown integrator"),
    };

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
        }
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
