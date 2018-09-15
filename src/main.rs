extern crate cgmath;
#[macro_use]
extern crate clap;
extern crate env_logger;
#[macro_use]
extern crate log;
extern crate rayon;
extern crate rustlight;

use clap::{App, Arg, SubCommand};
use rustlight::integrators::gradient::IntegratorGradient;
use rustlight::integrators::{Bitmap, Integrator};
use rustlight::scene::Scene;
use std::time::Instant;

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

enum IntegratorType {
    Primal(Box<Integrator>),
    Gradient(Box<IntegratorGradient>),
}
impl IntegratorType {
    fn compute(self, scene: &Scene) -> Bitmap {
        info!("Run Integrator...");
        let start = Instant::now();

        let img = match self {
            IntegratorType::Primal(mut v) => v.compute(scene),
            IntegratorType::Gradient(mut v) => {
                rustlight::integrators::gradient::IntegratorGradient::compute(&mut *v, scene)
            }
        };

        let elapsed = start.elapsed();
        info!(
            "Elapsed Integrator: {} ms",
            (elapsed.as_secs() * 1_000) + (elapsed.subsec_nanos() / 1_000_000) as u64
        );

        img
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
    let iterations_arg = Arg::with_name("iterations")
        .takes_value(true)
        .short("r")
        .default_value("50");
    let recons_type_arg = Arg::with_name("reconstruction_type")
        .takes_value(true)
        .short("t")
        .default_value("uniform");
    let matches =
        App::new("rustlight")
            .version("0.1.0")
            .author("Adrien Gruson <adrien.gruson@gmail.com>")
            .about("A Rusty Light Transport simulation program")
            .arg(
                Arg::with_name("scene")
                    .required(true)
                    .takes_value(true)
                    .index(1)
                    .help("JSON file description"),
            ).arg(Arg::with_name("average").short("a").takes_value(true).help(
                "average several pass of the integrator with a time limit ('inf' is possible)",
            )).arg(
                Arg::with_name("nbthreads")
                    .takes_value(true)
                    .short("t")
                    .default_value("auto")
                    .help("number of thread for the computation"),
            ).arg(
                Arg::with_name("image_scale")
                    .takes_value(true)
                    .short("s")
                    .default_value("1.0")
                    .help("image scaling factor"),
            ).arg(
                Arg::with_name("output")
                    .takes_value(true)
                    .short("o")
                    .help("output image file"),
            ).arg(Arg::with_name("debug").short("d").help("debug output"))
            .arg(
                Arg::with_name("nbsamples")
                    .short("n")
                    .takes_value(true)
                    .help("integration technique"),
            ).subcommand(
                SubCommand::with_name("path")
                    .about("path tracing")
                    .arg(&max_arg)
                    .arg(&min_arg),
            ).subcommand(
                SubCommand::with_name("gradient-path")
                    .about("gradient path tracing")
                    .arg(&max_arg)
                    .arg(&min_arg)
                    .arg(&iterations_arg)
                    .arg(&recons_type_arg),
            ).subcommand(
                SubCommand::with_name("gradient-path-explicit")
                    .about("gradient path tracing")
                    .arg(&max_arg)
                    .arg(&min_arg)
                    .arg(&iterations_arg)
                    .arg(&recons_type_arg)
                    .arg(
                        Arg::with_name("min_survival")
                            .takes_value(true)
                            .short("s")
                            .default_value("1.0"),
                    ),
            ).subcommand(
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
            ).subcommand(
                SubCommand::with_name("path-explicit")
                    .about("path tracing with explict light path construction")
                    .arg(&max_arg),
            ).subcommand(
                SubCommand::with_name("light-explicit")
                    .about("light tracing with explict light path construction")
                    .arg(&max_arg),
            ).subcommand(
                SubCommand::with_name("vpl")
                    .about("brute force virtual point light integrator")
                    .arg(&max_arg)
                    .arg(
                        Arg::with_name("clamping")
                            .takes_value(true)
                            .short("b")
                            .default_value("0.0"),
                    ).arg(
                        Arg::with_name("nb_vpl")
                            .takes_value(true)
                            .short("n")
                            .default_value("128"),
                    ),
            ).subcommand(
                SubCommand::with_name("ao").about("ambiant occlusion").arg(
                    Arg::with_name("distance")
                        .takes_value(true)
                        .short("d")
                        .default_value("inf"),
                ),
            ).subcommand(
                SubCommand::with_name("direct")
                    .about("direct lighting")
                    .arg(
                        Arg::with_name("bsdf")
                            .takes_value(true)
                            .short("b")
                            .default_value("1"),
                    ).arg(
                        Arg::with_name("light")
                            .takes_value(true)
                            .short("l")
                            .default_value("1"),
                    ),
            ).get_matches();

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

    ///////////////// Get the reconstruction algorithm
    let recons = match matches.subcommand() {
        ("gradient-path", Some(m)) | ("gradient-path-explicit", Some(m)) => {
            let iterations = value_t_or_exit!(m.value_of("iterations"), usize);
            let recons: Box<
                rustlight::integrators::gradient::PoissonReconstruction + Sync,
            > = match m.value_of("reconstruction_type").unwrap() {
                "uniform" => Box::new(
                    rustlight::integrators::gradient::recons::UniformPoissonReconstruction {
                        iterations,
                    },
                ),
                "weighted" => Box::new(
                    rustlight::integrators::gradient::recons::WeightedPoissonReconstruction::new(
                        iterations,
                    ),
                ),
                "bagging" => Box::new(
                    rustlight::integrators::gradient::recons::BaggingPoissonReconstruction {
                        iterations,
                        nb_buffers: if nb_samples <= 8 { nb_samples } else { 8 },
                    },
                ),
                _ => panic!("Impossible to found a reconstruction_type"),
            };
            Some(recons)
        }
        _ => None,
    };

    ///////////////// Create the main integrator
    let int = match matches.subcommand() {
        ("path-explicit", Some(m)) => {
            let max_depth = match_infinity(m.value_of("max").unwrap());
            IntegratorType::Primal(Box::new(
                rustlight::integrators::explicit::path::IntegratorPathTracing { max_depth },
            ))
        }
        ("light-explicit", Some(m)) => {
            let max_depth = match_infinity(m.value_of("max").unwrap());
            IntegratorType::Primal(Box::new(
                rustlight::integrators::explicit::light::IntegratorLightTracing { max_depth },
            ))
        }
        ("gradient-path", Some(m)) => {
            let max_depth = match_infinity(m.value_of("max").unwrap());
            let min_depth = match_infinity(m.value_of("min").unwrap());

            IntegratorType::Gradient(Box::new(
                rustlight::integrators::gradient::path::IntegratorGradientPath {
                    max_depth,
                    min_depth,
                    recons: recons.unwrap(),
                },
            ))
        }
        ("gradient-path-explicit", Some(m)) => {
            let max_depth = match_infinity(m.value_of("max").unwrap());
            let min_survival = value_t_or_exit!(m.value_of("min_survival"), f32);
            if min_survival <= 0.0 || min_survival > 1.0 {
                panic!("need to specify min_survival in ]0.0,1.0]");
            }
            IntegratorType::Gradient(Box::new(
                rustlight::integrators::gradient::explicit::IntegratorGradientPathTracing {
                    max_depth,
                    recons: recons.unwrap(),
                    min_survival: Some(min_survival),
                },
            ))
        }
        ("vpl", Some(m)) => {
            let max_depth = match_infinity(m.value_of("max").unwrap());
            let nb_vpl = value_t_or_exit!(m.value_of("nb_vpl"), usize);
            let clamping = value_t_or_exit!(m.value_of("clamping"), f32);
            IntegratorType::Primal(Box::new(
                rustlight::integrators::explicit::vpl::IntegratorVPL {
                    nb_vpl,
                    max_depth,
                    clamping_factor: if clamping <= 0.0 {
                        None
                    } else {
                        Some(clamping)
                    },
                },
            ))
        }
        ("path", Some(m)) => {
            let max_depth = match_infinity(m.value_of("max").unwrap());
            let min_depth = match_infinity(m.value_of("min").unwrap());
            IntegratorType::Primal(Box::new(rustlight::integrators::path::IntegratorPath {
                max_depth,
                min_depth,
            }))
        }
        ("pssmlt", Some(m)) => {
            let max_depth = match_infinity(m.value_of("max").unwrap());
            let min_depth = match_infinity(m.value_of("min").unwrap());
            let large_prob = value_t_or_exit!(m.value_of("large_prob"), f32);
            assert!(large_prob > 0.0 && large_prob <= 1.0);
            IntegratorType::Primal(Box::new(rustlight::integrators::pssmlt::IntegratorPSSMLT {
                large_prob,
                integrator: Box::new(rustlight::integrators::path::IntegratorPath {
                    max_depth,
                    min_depth,
                }),
            }))
        }
        ("ao", Some(m)) => {
            let dist = match_infinity(m.value_of("distance").unwrap());
            IntegratorType::Primal(Box::new(rustlight::integrators::ao::IntegratorAO {
                max_distance: dist,
            }))
        }
        ("direct", Some(m)) => {
            IntegratorType::Primal(Box::new(rustlight::integrators::direct::IntegratorDirect {
                nb_bsdf_samples: value_t_or_exit!(m.value_of("bsdf"), u32),
                nb_light_samples: value_t_or_exit!(m.value_of("light"), u32),
            }))
        }
        _ => panic!("unknown integrator"),
    };
    let int = if matches.is_present("average") {
        let time_out = match_infinity(matches.value_of("average").unwrap());
        let int = match int {
            IntegratorType::Gradient(v) => IntegratorType::Primal(Box::new(
                rustlight::integrators::gradient::avg::IntegratorGradientAverage {
                    time_out,
                    output_csv: false,
                    integrator: v,
                },
            )),
            IntegratorType::Primal(v) => {
                IntegratorType::Primal(Box::new(rustlight::integrators::avg::IntegratorAverage {
                    time_out,
                    output_csv: false,
                    integrator: v,
                }))
            }
        };
        int
    } else {
        int
    };
    let img = int.compute(&scene);

    // Save the image
    rustlight::tools::save(imgout_path_str, &img, "primal");
}
