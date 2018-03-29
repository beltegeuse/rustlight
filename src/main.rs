extern crate byteorder;
extern crate cgmath;
#[macro_use]
extern crate clap;
extern crate env_logger;
extern crate image;
#[macro_use]
extern crate log;
extern crate rayon;
extern crate rustlight;

use byteorder::{LittleEndian, WriteBytesExt};
use cgmath::Point2;
use clap::{App, Arg, SubCommand};
use image::*;
use rustlight::integrator::ColorGradient;
use rustlight::Scale;
use rustlight::scene::Bitmap;
use rustlight::structure::Color;
use std::io::prelude::*;
use std::time::Instant;

fn save_pfm(imgout_path_str: &str, img: &Bitmap<Color>) {
    let mut file = std::fs::File::create(std::path::Path::new(imgout_path_str)).unwrap();
    let header = format!("PF\n{} {}\n-1.0\n", img.size.y, img.size.x);
    file.write(header.as_bytes()).unwrap();
    for y in 0..img.size.y {
        for x in 0..img.size.x {
            let p = img.get(Point2::new(img.size.x - x - 1, img.size.y - y - 1));
            file.write_f32::<LittleEndian>(p.r.abs()).unwrap();
            file.write_f32::<LittleEndian>(p.g.abs()).unwrap();
            file.write_f32::<LittleEndian>(p.b.abs()).unwrap();
        }
    }
}

fn save_png(imgout_path_str: &str, img: &Bitmap<Color>) {
    // The image that we will render
    let mut image_ldr = DynamicImage::new_rgb8(img.size.x, img.size.y);
    for x in 0..img.size.x {
        for y in 0..img.size.y {
            image_ldr.put_pixel(x, y, img.get(Point2::new(img.size.x - x - 1, y)).to_rgba())
        }
    }
    let ref mut fout = std::fs::File::create(imgout_path_str).unwrap();
    image_ldr
        .save(fout, image::PNG)
        .expect("failed to write img into file");
}

fn classical_mc_integration(
    scene: &rustlight::scene::Scene,
    nb_samples: u32,
    nb_threads: Option<usize>,
    int: Box<rustlight::integrator::Integrator<Color> + Sync + Send>,
) -> Bitmap<Color> {
    ////////////// Do the rendering
    info!("Rendering...");
    let start = Instant::now();
    let pool = match nb_threads {
        None => rayon::ThreadPoolBuilder::new(),
        Some(x) => rayon::ThreadPoolBuilder::new().num_threads(x),
    }.build()
        .unwrap();
    let img = pool.install(|| scene.render(int, nb_samples));
    let elapsed = start.elapsed();
    info!(
        "Elapsed: {} ms",
        (elapsed.as_secs() * 1_000) + (elapsed.subsec_nanos() / 1_000_000) as u64
    );

    return img;
}

fn gradient_domain_integration(
    scene: &rustlight::scene::Scene,
    nb_samples: u32,
    nb_threads: Option<usize>,
    int: Box<rustlight::integrator::Integrator<ColorGradient> + Sync + Send>,
) -> Bitmap<Color> {
    info!("Rendering...");
    let start = Instant::now();
    let pool = match nb_threads {
        None => rayon::ThreadPoolBuilder::new(),
        Some(x) => rayon::ThreadPoolBuilder::new().num_threads(x),
    }.build()
        .unwrap();
    let img_grad = pool.install(|| scene.render(int, nb_samples));
    let elapsed = start.elapsed();
    info!(
        "Elapsed: {} ms",
        (elapsed.as_secs() * 1_000) + (elapsed.subsec_nanos() / 1_000_000) as u64
    );

    // Generates images buffers (dx, dy, primal)
    let mut primal_image: Bitmap<Color> = Bitmap::new(Point2::new(0, 0), *scene.camera.size());
    let mut dx_image = Bitmap::new(Point2::new(0, 0), *scene.camera.size());
    let mut dy_image = Bitmap::new(Point2::new(0, 0), *scene.camera.size());
    for y in 0..img_grad.size.y {
        for x in 0..img_grad.size.x {
            let pos = Point2::new(x, y);
            let curr = img_grad.get(pos);
            primal_image.accumulate(pos, &curr.main);
            for (i, off) in rustlight::integrator::GRADIENT_ORDER.iter().enumerate() {
                let pos_off: Point2<i32> = Point2::new(pos.x as i32 + off.x, pos.y as i32 + off.y);
                primal_image.accumulate_safe(pos_off, curr.radiances[i].clone());
                match rustlight::integrator::GRADIENT_DIRECTION[i] {
                    rustlight::integrator::GradientDirection::X(v) => match v {
                        1 => dx_image.accumulate(pos, &curr.gradients[i]),
                        -1 => dx_image.accumulate_safe(pos_off, curr.gradients[i].clone() * -1.0),
                        _ => panic!("wrong displacement X"), // FIXME: Fix the enum
                    },
                    rustlight::integrator::GradientDirection::Y(v) => match v {
                        1 => dy_image.accumulate(pos, &curr.gradients[i]),
                        -1 => dy_image.accumulate_safe(pos_off, curr.gradients[i].clone() * -1.0),
                        _ => panic!("wrong displacement Y"),
                    },
                }
            }
        }
    }
    // Scale the throughtput image
    primal_image.scale(1.0 / 4.0); // TODO: Wrong at the corners, need to fix it

    // Output the images
    // FIXME: Add the ability to output images
    /*{
        save_pfm("out_primal.pfm", &primal_image);
        save_pfm("out_dx.pfm", &dx_image);
        save_pfm("out_dy.pfm", &dy_image);
    }*/

    info!("Reconstruction...");
    let start = Instant::now();
    // Reconstruction (image-space covariate, uniform reconstruction)
    let mut current: Box<Bitmap<Color>> =
        Box::new(Bitmap::new(Point2::new(0, 0), scene.camera.size().clone()));
    let mut next: Box<Bitmap<Color>> =
        Box::new(Bitmap::new(Point2::new(0, 0), scene.camera.size().clone()));
    // 1) Init
    for y in 0..img_grad.size.y {
        for x in 0..img_grad.size.x {
            let pos = Point2::new(x, y);
            current.accumulate(pos, primal_image.get(pos));
        }
    }
    for _iter in 0..50 {
        // FIXME: Do it multi-threaded
        next.reset(); // Reset all to black
        for y in 0..img_grad.size.y {
            for x in 0..img_grad.size.x {
                let pos = Point2::new(x, y);
                let mut c = current.get(pos).clone();
                let mut w = 1.0;
                if x > 0 {
                    let pos_off = Point2::new(x - 1, y);
                    c += current.get(pos_off).clone() + dx_image.get(pos_off).clone();
                    w += 1.0;
                }
                if x < img_grad.size.x - 1 {
                    let pos_off = Point2::new(x + 1, y);
                    c += current.get(pos_off).clone() - dx_image.get(pos).clone();
                    w += 1.0;
                }
                if y > 0 {
                    let pos_off = Point2::new(x, y - 1);
                    c += current.get(pos_off).clone() + dy_image.get(pos_off).clone();
                    w += 1.0;
                }
                if y < img_grad.size.y - 1 {
                    let pos_off = Point2::new(x, y + 1);
                    c += current.get(pos_off).clone() - dy_image.get(pos).clone();
                    w += 1.0;
                }
                c.scale(1.0 / w);
                next.accumulate(pos, &c);
            }
        }
        std::mem::swap(&mut current, &mut next);
    }
    let elapsed = start.elapsed();
    info!(
        "Elapsed: {} ms",
        (elapsed.as_secs() * 1_000) + (elapsed.subsec_nanos() / 1_000_000) as u64
    );

    // Export the reconstruction
    let mut image: Bitmap<Color> = Bitmap::new(Point2::new(0, 0), scene.camera.size().clone());
    for x in 0..img_grad.size.x {
        for y in 0..img_grad.size.y {
            let pos = Point2::new(x, y);
            let pix_value = next.get(pos).clone() + img_grad.get(pos).very_direct.clone();
            image.accumulate(pos, &pix_value);
        }
    }
    image
}

fn match_infinity<T: std::str::FromStr>(input: &str) -> Option<T> {
    match input {
        "inf" => None,
        _ => match input.parse::<T>() {
            Ok(x) => Some(x),
            Err(_e) => panic!("wrong input for inf type parameter"),
        },
    }
}

fn main() {
    // Read input args
    let matches = App::new("rustlight")
        .version("0.0.2")
        .author("Adrien Gruson <adrien.gruson@gmail.com>")
        .about("A Rusty Light Transport simulation program")
        .arg(
            Arg::with_name("scene")
                .required(true)
                .takes_value(true)
                .index(1)
                .help("JSON file description"),
        )
        .arg(
            Arg::with_name("nbthreads")
                .takes_value(true)
                .short("t")
                .default_value("auto")
                .help("number of thread for the computation"),
        )
        .arg(
            Arg::with_name("output")
                .takes_value(true)
                .short("o")
                .help("output image file"),
        )
        .arg(
            Arg::with_name("nbsamples")
                .short("n")
                .takes_value(true)
                .help("integration technique"),
        )
        .subcommand(
            SubCommand::with_name("path")
                .about("path tracing")
                .arg(
                    Arg::with_name("max")
                        .takes_value(true)
                        .short("m")
                        .default_value("inf"),
                )
                .arg(
                    Arg::with_name("min")
                        .takes_value(true)
                        .short("n")
                        .default_value("inf"),
                ),
        )
        .subcommand(
            SubCommand::with_name("path-explicit")
                .about("path tracing with explict light path construction")
                .arg(
                    Arg::with_name("max")
                        .takes_value(true)
                        .short("m")
                        .default_value("inf"),
                ),
        )
        .subcommand(
            SubCommand::with_name("gd-path")
                .about("gradient-domain path tracing")
                .arg(
                    Arg::with_name("max")
                        .takes_value(true)
                        .short("m")
                        .default_value("inf"),
                )
                .arg(
                    Arg::with_name("min")
                        .takes_value(true)
                        .short("n")
                        .default_value("inf"),
                ),
        )
        .subcommand(
            SubCommand::with_name("ao").about("ambiant occlusion").arg(
                Arg::with_name("distance")
                    .takes_value(true)
                    .short("d")
                    .default_value("inf"),
            ),
        )
        .subcommand(
            SubCommand::with_name("direct")
                .about("direct lighting")
                .arg(
                    Arg::with_name("bsdf")
                        .takes_value(true)
                        .short("b")
                        .default_value("1"),
                )
                .arg(
                    Arg::with_name("light")
                        .takes_value(true)
                        .short("l")
                        .default_value("1"),
                ),
        )
        .get_matches();

    /////////////// Setup logging system
    env_logger::Builder::from_default_env()
        .default_format_timestamp(false)
        .parse("info")
        .init();

    /////////////// Check output extension
    let imgout_path_str = matches.value_of("output").unwrap_or("test.pfm");
    let output_ext = match std::path::Path::new(imgout_path_str).extension() {
        None => panic!("No file extension provided"),
        Some(x) => std::ffi::OsStr::to_str(x).expect("Issue to unpack the file"),
    };

    //////////////// Load the rendering configuration
    let nb_samples = value_t_or_exit!(matches.value_of("nbsamples"), u32);
    let nb_threads = match matches.value_of("nbthreads").unwrap() {
        "auto" => None,
        x => {
            let v = x.parse::<usize>().expect("Wrong number of thread");
            if v == 0 {
                panic!("Impossible to use 0 thread for the computation");
            }
            Some(v)
        }
    };

    //////////////// Load the scene
    let scene_path_str = matches
        .value_of("scene")
        .expect("no scene parameter provided");
    let scene_path = std::path::Path::new(scene_path_str);
    // - read the file
    let mut fscene = std::fs::File::open(scene_path).expect("scene file not found");
    let mut data = String::new();
    fscene
        .read_to_string(&mut data)
        .expect("impossible to read the file");
    // - build the scene
    let wk = scene_path
        .parent()
        .expect("impossible to extract parent directory for OBJ loading");
    let scene = rustlight::scene::Scene::new(&data, wk).expect("error when loading the scene");

    let img = match matches.subcommand() {
        ("path-explicit", Some(m)) => {
            let max_depth = match_infinity(m.value_of("max").unwrap());

            classical_mc_integration(
                &scene,
                nb_samples,
                nb_threads,
                Box::new(rustlight::integrator::IntegratorUniPath { max_depth }),
            )
        }
        ("path", Some(m)) => {
            let max_depth = match_infinity(m.value_of("max").unwrap());
            let min_depth = match_infinity(m.value_of("min").unwrap());

            classical_mc_integration(
                &scene,
                nb_samples,
                nb_threads,
                Box::new(rustlight::integrator::IntegratorPath {
                    max_depth,
                    min_depth,
                }),
            )
        }
        ("gd-path", Some(m)) => {
            let max_depth = match_infinity(m.value_of("max").unwrap());
            let min_depth = match_infinity(m.value_of("min").unwrap());

            gradient_domain_integration(
                &scene,
                nb_samples,
                nb_threads,
                Box::new(rustlight::integrator::IntegratorPath {
                    max_depth,
                    min_depth,
                }),
            )
        }
        ("ao", Some(m)) => {
            let dist = match_infinity(m.value_of("distance").unwrap());

            classical_mc_integration(
                &scene,
                nb_samples,
                nb_threads,
                Box::new(rustlight::integrator::IntegratorAO { max_distance: dist }),
            )
        }
        ("direct", Some(m)) => classical_mc_integration(
            &scene,
            nb_samples,
            nb_threads,
            Box::new(rustlight::integrator::IntegratorDirect {
                nb_bsdf_samples: value_t_or_exit!(m.value_of("bsdf"), u32),
                nb_light_samples: value_t_or_exit!(m.value_of("light"), u32),
            }),
        ),
        _ => panic!("unknown integrator"),
    };

    // Save the image (HDR and LDF)
    // -- LDR
    match output_ext {
        "pfm" => {
            save_pfm(imgout_path_str, &img);
        }
        "png" => {
            save_png(imgout_path_str, &img);
        }
        _ => panic!("Unknow output file extension"),
    }
}
