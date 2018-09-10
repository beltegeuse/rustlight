extern crate cgmath;
#[macro_use]
extern crate clap;
extern crate env_logger;
#[macro_use]
extern crate log;
extern crate rayon;
extern crate rustlight;

use clap::{App, Arg, SubCommand};
use std::io::Read;

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
    let max_arg = Arg::with_name("max")
        .takes_value(true)
        .short("m")
        .default_value("inf");
    let min_arg = Arg::with_name("min")
        .takes_value(true)
        .short("n")
        .default_value("inf");
    let _iterations_arg = Arg::with_name("iterations")
        .takes_value(true)
        .short("r")
        .default_value("50");
    let matches = App::new("rustlight")
        .version("0.0.5")
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
            Arg::with_name("average")
                .short("a")
                .takes_value(true)
                .default_value("inf")
                .help("average several pass of the integrator with a time limit"),
        )
        .arg(
            Arg::with_name("nbthreads")
                .takes_value(true)
                .short("t")
                .default_value("auto")
                .help("number of thread for the computation"),
        )
        .arg(
            Arg::with_name("image_scale")
                .takes_value(true)
                .short("s")
                .default_value("1.0")
                .help("image scaling factor"),
        )
        .arg(
            Arg::with_name("output")
                .takes_value(true)
                .short("o")
                .help("output image file"),
        )
        .arg(Arg::with_name("debug").short("d").help("debug output"))
        .arg(
            Arg::with_name("nbsamples")
                .short("n")
                .takes_value(true)
                .help("integration technique"),
        )
        .subcommand(
            SubCommand::with_name("path")
                .about("path tracing")
                .arg(&max_arg)
                .arg(&min_arg),
        )
        .subcommand(
            SubCommand::with_name("pssmlt")
                .about("path tracing with MCMC sampling")
                .arg(&max_arg)
                .arg(&min_arg)
                .arg(
                    Arg::with_name("large_prob")
                        .takes_value(true)
                        .short("p")
                        .default_value("0.3"),
                ),
        )
        .subcommand(
            SubCommand::with_name("path-explicit")
                .about("path tracing with explict light path construction")
                .arg(&max_arg),
        )
        .subcommand(
            SubCommand::with_name("light-explicit")
                .about("light tracing with explict light path construction")
                .arg(&max_arg),
        )
        .subcommand(
            SubCommand::with_name("vpl")
                .about("brute force virtual point light integrator")
                .arg(&max_arg)
                .arg(
                    Arg::with_name("clamping")
                        .takes_value(true)
                        .short("b")
                        .default_value("0.0"),
                )
                .arg(
                    Arg::with_name("nb_vpl")
                        .takes_value(true)
                        .short("n")
                        .default_value("128"),
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
    if matches.is_present("debug") {
        // FIXME: add debug flag?
        env_logger::Builder::from_default_env()
            .default_format_timestamp(false)
            .init();
    } else {
        env_logger::Builder::from_default_env()
            .default_format_timestamp(false)
            .parse("info")
            .init();
    }
    /////////////// Check output extension
    let imgout_path_str = matches.value_of("output").unwrap_or("test.pfm");
    let output_ext = match std::path::Path::new(imgout_path_str).extension() {
        None => panic!("No file extension provided"),
        Some(x) => std::ffi::OsStr::to_str(x).expect("Issue to unpack the file"),
    };

    //////////////// Load the rendering configuration
    let nb_samples = value_t_or_exit!(matches.value_of("nbsamples"), usize);
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
    let mut scene = rustlight::scene::Scene::new(
        &data,
        wk,
        nb_samples,
        nb_threads,
        imgout_path_str.to_string(),
    ).expect("error when loading the scene");

    ///////////////// Tweak the image size
    {
        let image_scale = value_t_or_exit!(matches.value_of("image_scale"), f32);
        if image_scale != 1.0 {
            info!("Scale the image: {:?}", image_scale);
            assert!(image_scale != 0.0);
            scene.camera.scale_image(image_scale);
        }
    }

    ///////////////// Create the main integrator
    let mut int: Box<rustlight::integrators::Integrator> = match matches.subcommand() {
        ("path-explicit", Some(m)) => {
            let max_depth = match_infinity(m.value_of("max").unwrap());
            Box::new(rustlight::integrators::explicit::path::IntegratorPathTracing { max_depth })
        }
        ("light-explicit", Some(m)) => {
            let max_depth = match_infinity(m.value_of("max").unwrap());
            Box::new(rustlight::integrators::explicit::light::IntegratorLightTracing { max_depth })
        }
        ("vpl", Some(m)) => {
            let max_depth = match_infinity(m.value_of("max").unwrap());
            let nb_vpl = value_t_or_exit!(m.value_of("nb_vpl"), usize);
            let clamping = value_t_or_exit!(m.value_of("clamping"), f32);
            Box::new(rustlight::integrators::explicit::vpl::IntegratorVPL {
                nb_vpl,
                max_depth,
                clamping_factor: if clamping <= 0.0 {
                    None
                } else {
                    Some(clamping)
                },
            })
        }
        ("path", Some(m)) => {
            let max_depth = match_infinity(m.value_of("max").unwrap());
            let min_depth = match_infinity(m.value_of("min").unwrap());
            Box::new(rustlight::integrators::path::IntegratorPath {
                max_depth,
                min_depth,
            })
        }
        ("pssmlt", Some(m)) => {
            let max_depth = match_infinity(m.value_of("max").unwrap());
            let min_depth = match_infinity(m.value_of("min").unwrap());
            let large_prob = value_t_or_exit!(m.value_of("large_prob"), f32);
            assert!(large_prob > 0.0 && large_prob <= 1.0);
            Box::new(rustlight::integrators::pssmlt::IntegratorPSSMLT {
                large_prob,
                integrator: Box::new(rustlight::integrators::path::IntegratorPath {
                    max_depth,
                    min_depth,
                }),
            })
        }
        ("ao", Some(m)) => {
            let dist = match_infinity(m.value_of("distance").unwrap());
            Box::new(rustlight::integrators::ao::IntegratorAO { max_distance: dist })
        }
        ("direct", Some(m)) => Box::new(rustlight::integrators::direct::IntegratorDirect {
            nb_bsdf_samples: value_t_or_exit!(m.value_of("bsdf"), u32),
            nb_light_samples: value_t_or_exit!(m.value_of("light"), u32),
        }),
        _ => panic!("unknown integrator"),
    };
    if matches.is_present("average") {
        let time_out = match_infinity(matches.value_of("average").unwrap());
        int = Box::new(rustlight::integrators::avg::IntegratorAverage {
            time_out,
            output_csv: false,
            integrator: int,
        })
    }
    let img = rustlight::integrators::run_integrator(&scene, int.as_mut());

    // Save the image (HDR and LDF)
    // -- LDR
    match output_ext {
        "pfm" => {
            rustlight::tools::save_pfm(imgout_path_str, &img, &"primal".to_string());
        }
        "png" => {
            rustlight::tools::save_png(imgout_path_str, &img, &"primal".to_string());
        }
        _ => panic!("Unknow output file extension"),
    }
}
