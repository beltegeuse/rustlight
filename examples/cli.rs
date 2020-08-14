#![cfg_attr(feature = "clippy", feature(plugin))]
#![cfg_attr(feature = "clippy", plugin(clippy))]
#![allow(dead_code)]
#![allow(clippy::float_cmp)]
#![allow(clippy::cognitive_complexity)]

extern crate cgmath;
extern crate num_cpus;
#[macro_use]
extern crate clap;
extern crate env_logger;
#[macro_use]
extern crate log;
extern crate rand;
extern crate rayon;
extern crate rustlight;

use clap::{App, Arg, SubCommand};
use rand::SeedableRng;
use rustlight::integrators::IntegratorType;
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
        .help("max path depth")
        .default_value("inf");
    let min_arg = Arg::with_name("min")
        .takes_value(true)
        .short("n")
        .help("min path depth")
        .default_value("inf");
    let iterations_arg = Arg::with_name("iterations")
        .takes_value(true)
        .short("r")
        .help("number of iteration used to reconstruct an image")
        .default_value("50");
    let recons_type_arg = Arg::with_name("reconstruction_type")
        .takes_value(true)
        .short("t")
        .default_value("uniform");
    let matches =
        App::new("rustlight")
            .version("0.2.0")
            .author("Adrien Gruson <adrien.gruson@gmail.com>")
            .about("A Rusty Light Transport simulation program")
            .arg(
                Arg::with_name("scene")
                    .required(true)
                    .takes_value(true)
                    .index(1)
                    .help("JSON/PBRT file path (scene description)"),
            )
            .arg(Arg::with_name("average").short("a").takes_value(true).help(
                "average several pass of the integrator with a time limit ('inf' is possible)",
            ))
            .arg(
                Arg::with_name("nbthreads")
                    .takes_value(true)
                    .allow_hyphen_values(true)
                    .short("t")
                    .default_value("auto")
                    .help("number of thread for the computation (could be negative)"),
            )
            .arg(
                Arg::with_name("random_number_generator")
                .takes_value(true)
                .short("r")
                .default_value("independent")
                .help("the random number generator used"),
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
            .arg(
                Arg::with_name("medium")
                    .short("m")
                    .takes_value(true)
                    .default_value("0.0")
                    .help("add medium with defined density"),
            )
            .arg(Arg::with_name("debug").short("d").help("debug output"))
            .arg(
                Arg::with_name("nbsamples")
                    .short("n")
                    .takes_value(true)
                    .help("number of sample from the sensor (if applicable)"),
            )
            .subcommand(
                SubCommand::with_name("gradient-path")
                    .about("gradient path tracing")
                    .arg(&max_arg)
                    .arg(&min_arg)
                    .arg(&iterations_arg)
                    .arg(&recons_type_arg),
            )
            .subcommand(
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
            )
            .subcommand(
                SubCommand::with_name("pssmlt")
                    .about("path tracing with MCMC sampling")
                    .arg(&max_arg)
                    .arg(&min_arg)
                    .arg(
                        Arg::with_name("strategy")
                            .takes_value(true)
                            .short("s")
                            .help("different sampling strategy: [all, bsdf, emitter]")
                            .default_value("all"),
                    )
                    .arg(
                        Arg::with_name("large_prob")
                            .help("probability to perform a large step")
                            .takes_value(true)
                            .short("p")
                            .default_value("0.3"),
                    )
                    .arg(
                        Arg::with_name("nb_samples_norm")
                            .help("number of samples to compute to sample X0")
                            .takes_value(true)
                            .short("b")
                            .default_value("100000"),
                    ),
            )
            .subcommand(
                SubCommand::with_name("smcmc")
                    .about("Stratified MCMC")
                    .arg(&max_arg)
                    .arg(&min_arg)
                    .arg(
                        Arg::with_name("strategy")
                            .takes_value(true)
                            .short("s")
                            .help("different sampling strategy: [all, bsdf, emitter]")
                            .default_value("all"),
                    )
                    .arg(
                        Arg::with_name("large_prob")
                            .help("probability to perform a large step")
                            .takes_value(true)
                            .short("p")
                            .default_value("0.3"),
                    ).arg(
                        Arg::with_name("recons")
                            .takes_value(true)
                            .short("r")
                            .help("recons image: [naive,irls]")
                            .default_value("irls"),
                    )
                    .arg(
                        Arg::with_name("init")
                            .takes_value(true)
                            .short("i")
                            .help("init algorithm: [independent,mcmc]")
                            .default_value("mcmc"),
                    )
            )
            .subcommand(
                SubCommand::with_name("erpt")
                    .about("path tracing with MCMC sampling")
                    .arg(&max_arg)
                    .arg(&min_arg)
                    .arg(
                        Arg::with_name("no_stratified")
                        .short("k")
                        .help("remove stratification of ERPT")
                    )
                    .arg(
                        Arg::with_name("strategy")
                            .takes_value(true)
                            .short("s")
                            .help("difefrent sampling strategy: [all, bsdf, emitter]")
                            .default_value("all"),
                    )
                    .arg(
                        Arg::with_name("nb_mc")
                            .takes_value(true)
                            .short("e")
                            .help("number of MC samples")
                            .default_value("1"),
                    ).arg(
                        Arg::with_name("chain_samples")
                            .takes_value(true)
                            .short("c")
                            .help("number of chains samples")
                            .default_value("100"),
                    ),
            )
            .subcommand(
                SubCommand::with_name("path_kulla")
                    .about("path tracing for single scattering")
                    .arg(
                        Arg::with_name("strategy")
                            .takes_value(true)
                            .short("s")
                            .help("different sampling strategy: [all, kulla_position, transmittance_phase]")
                            .default_value("all"),
                    )
            )
            .subcommand(
                SubCommand::with_name("path")
                    .about("path tracing generating path from the sensor")
                    .arg(&max_arg)
                    .arg(&min_arg)
                    .arg(
                        Arg::with_name("strategy")
                            .takes_value(true)
                            .short("s")
                            .help("difefrent sampling strategy: [all, bsdf, emitter]")
                            .default_value("all"),
                    ).arg(
                        Arg::with_name("single")
                            .help("to only compute single scattering")
                            .short("x"),
                    ),
            )
            .subcommand(
                SubCommand::with_name("light")
                    .about("light tracing generating path from the lights")
                    .arg(&max_arg)
                    .arg(&min_arg)
                    .arg(
                        Arg::with_name("lightpaths")
                            .takes_value(true)
                            .help("number of light path generated from the light sources")
                            .short("p")
                            .default_value("all"),
                    ),
            )
            .subcommand(
                SubCommand::with_name("vpl")
                    .about("brute force virtual point light integrator")
                    .arg(&max_arg)
                    .arg(
                        Arg::with_name("clamping")
                            .takes_value(true)
                            .short("b")
                            .help("clamping factor")
                            .default_value("0.0"),
                    )
                    .arg(
                        Arg::with_name("nb_vpl")
                            .takes_value(true)
                            .help("number of VPL at least generated")
                            .short("n")
                            .default_value("128"),
                    )
                    .arg(
                        Arg::with_name("option_lt")
                        .takes_value(true)
                        .help("option to select light transport: [all, surface, volume]")
                        .short("l")
                        .default_value("all")
                    )
                    .arg(
                        Arg::with_name("option_vpl")
                        .takes_value(true)
                        .help("option to select generated VPL: [all, surface, volume]")
                        .short("v")
                        .default_value("all")
                    ),
            )
            .subcommand(
                SubCommand::with_name("vol_primitives")
                    .about("BRE/Beam/Planes estimators")
                    .arg(&max_arg)
                    .arg(
                        Arg::with_name("nb_primitive")
                            .takes_value(true)
                            .help("number of primitive generated")
                            .short("n")
                            .default_value("128"),
                    )
                    .arg(
                        Arg::with_name("primitives")
                            .takes_value(true)
                            .help("type of primitives: [beam, bre, planes, vrl]")
                            .short("p")
                            .default_value("bre"),
                    ),
            )
            .subcommand(
                SubCommand::with_name("plane_single")
                    .about("Prototype implementation of 'Photon surfaces for robust, unbiased volumetric density estimation'")
                    .arg(
                        Arg::with_name("nb_primitive")
                            .takes_value(true)
                            .help("number of primitive generated")
                            .short("n")
                            .default_value("128"),
                    )
                    .arg(
                        Arg::with_name("strategy")
                            .takes_value(true)
                            .help("sampling strategy: [uv, vt, st, cmis, dmis, average]")
                            .short("s")
                            .default_value("average"),
                    ),
            )
            .subcommand(
                SubCommand::with_name("uncorrelated_plane_single")
                    .about("Prototype implementation of 'Photon surfaces for robust, unbiased volumetric density estimation'")
                    .arg(
                        Arg::with_name("nb_primitive")
                            .takes_value(true)
                            .help("number of primitive generated (per pixel)")
                            .short("n")
                            .default_value("128"),
                    )
                    .arg(
                        Arg::with_name("strategy")
                            .takes_value(true)
                            .short("s")
                            .help("sampling strategy: [uv, vt, st, cmis, dmis, average]")
                            .default_value("average"),
                    ),
            )
            .subcommand(
                SubCommand::with_name("ao")
                    .about("ambiant occlusion")
                    .arg(
                        Arg::with_name("distance")
                            .takes_value(true)
                            .short("d")
                            .help("distance threshold for AO")
                            .default_value("inf"),
                    )
                    .arg(
                        Arg::with_name("normal-correction")
                            .takes_value(false)
                            .help("apply normal correction")
                            .short("n"),
                    ),
            )
            .subcommand(
                SubCommand::with_name("direct")
                    .about("direct lighting")
                    .arg(
                        Arg::with_name("bsdf")
                            .takes_value(true)
                            .help("number of samples from the BSDF")
                            .short("b")
                            .default_value("1"),
                    )
                    .arg(
                        Arg::with_name("light")
                            .takes_value(true)
                            .help("number of samples from the emitter")
                            .short("l")
                            .default_value("1"),
                    ),
            )
            .get_matches();

    /////////////// Setup logging system
    if matches.is_present("debug") {
        // FIXME: add debug flag?
        env_logger::Builder::from_default_env()
            .format_timestamp(None)
            .init();
    } else {
        env_logger::Builder::from_default_env()
            .format_timestamp(None)
            .parse_filters("info")
            .init();
    }
    /////////////// Check output extension
    let imgout_path_str = matches.value_of("output").unwrap_or("test.pfm");

    //////////////// Load the rendering configuration
    let nb_samples = value_t_or_exit!(matches.value_of("nbsamples"), usize);

    //////////////// Load the scene
    let scene = matches
        .value_of("scene")
        .expect("no scene parameter provided");
    let scene = rustlight::scene_loader::SceneLoaderManager::default()
        .load(scene.to_string())
        .expect("error on loading the scene");
    let scene = match matches.value_of("nbthreads").unwrap() {
        "auto" => scene,
        x => {
            let v = x.parse::<i32>().expect("Wrong number of thread");
            match v {
                v if v > 0 => scene.nb_threads(v as usize),
                v if v < 0 => {
                    let nb_threads = num_cpus::get() as i32 + v;
                    if nb_threads < 0 {
                        panic!("Not enough threads: {} removing {}", num_cpus::get(), v);
                    }
                    info!("Run with {} threads", nb_threads);
                    scene.nb_threads(nb_threads as usize)
                }
                _ => {
                    panic!("Impossible to use 0 thread for the computation");
                }
            }
        }
    };
    let mut scene = scene.nb_samples(nb_samples).output_img(imgout_path_str);

    ///////////////// Medium
    // TODO: Read from PBRT file
    {
        let medium_density = value_t_or_exit!(matches.value_of("medium"), f32);
        if medium_density != 0.0 {
            let sigma_a = rustlight::structure::Color::value(0.0) * medium_density;
            let sigma_s = rustlight::structure::Color::value(1.0) * medium_density;
            let sigma_t = sigma_a + sigma_s;
            scene.volume = Some(rustlight::volume::HomogenousVolume {
                sigma_a,
                sigma_s,
                sigma_t,
                density: 1.0,
            });

            info!("Create volume with: ");
            info!(" - sigma_a: {:?}", sigma_a);
            info!(" - sigma_s: {:?}", sigma_s);
            info!(" - sigma_t: {:?}", sigma_t);
        }
    }
    ///////////////// Tweak the image size
    {
        let image_scale = value_t_or_exit!(matches.value_of("image_scale"), f32);
        if image_scale != 1.0 {
            info!("Scale the image: {:?}", image_scale);
            assert!(image_scale != 0.0);
            scene.camera.scale_image(image_scale);
        }
    }

    // Build internal
    scene.build_emitters();

    ///////////////// Get the reconstruction algorithm
    let recons = match matches.subcommand() {
        ("gradient-path", Some(m)) | ("gradient-path-explicit", Some(m)) => {
            let iterations = value_t_or_exit!(m.value_of("iterations"), usize);
            let recons: Box<dyn rustlight::integrators::PoissonReconstruction + Sync> = match m
                .value_of("reconstruction_type")
                .unwrap()
            {
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
    let mut int = match matches.subcommand() {
        ("path_kulla", Some(m)) => {
            let strategy = value_t_or_exit!(m.value_of("strategy"), String);
            let strategy = match strategy.as_ref() {
                "all" => {
                    rustlight::integrators::explicit::path_kulla::IntegratorPathKullaStrategies::All
                }
                "kulla_position" => {
                    rustlight::integrators::explicit::path_kulla::IntegratorPathKullaStrategies::KullaPosition
                }
                "transmittance_phase" => {
                    rustlight::integrators::explicit::path_kulla::IntegratorPathKullaStrategies::TransmittancePhase
                }
                _ => panic!("invalid strategy: {}", strategy),
            };
            IntegratorType::Primal(Box::new(
                rustlight::integrators::explicit::path_kulla::IntegratorPathKulla { strategy },
            ))
        }
        ("path", Some(m)) => {
            let single_scattering = m.is_present("single");
            let max_depth = match_infinity(m.value_of("max").unwrap());
            let min_depth = match_infinity(m.value_of("min").unwrap());
            let strategy = value_t_or_exit!(m.value_of("strategy"), String);
            let strategy = match strategy.as_ref() {
                "all" => {
                    rustlight::integrators::explicit::path::IntegratorPathTracingStrategies::All
                }
                "bsdf" => {
                    rustlight::integrators::explicit::path::IntegratorPathTracingStrategies::BSDF
                }
                "emitter" => {
                    rustlight::integrators::explicit::path::IntegratorPathTracingStrategies::Emitter
                }
                _ => panic!("invalid strategy: {}", strategy),
            };
            IntegratorType::Primal(Box::new(
                rustlight::integrators::explicit::path::IntegratorPathTracing {
                    min_depth,
                    max_depth,
                    strategy,
                    single_scattering,
                },
            ))
        }
        ("light", Some(m)) => {
            let max_depth = match_infinity(m.value_of("max").unwrap());
            let min_depth = match_infinity(m.value_of("min").unwrap());
            let strategy = value_t_or_exit!(m.value_of("lightpaths"), String);
            let (render_surface, render_volume) = match strategy.as_ref() {
                "all" => (true, true),
                "surface" => (true, false),
                "volume" => (false, true),
                _ => panic!("invalid lightpaths type to render"),
            };
            IntegratorType::Primal(Box::new(
                rustlight::integrators::explicit::light::IntegratorLightTracing {
                    max_depth,
                    min_depth,
                    render_surface,
                    render_volume,
                },
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
            let get_option = |name: &'static str| {
                let options = value_t_or_exit!(m.value_of(name), String);
                match options.as_ref() {
                    "all" => rustlight::integrators::explicit::vpl::IntegratorVPLOption::All,
                    "surface" => {
                        rustlight::integrators::explicit::vpl::IntegratorVPLOption::Surface
                    }
                    "volume" => rustlight::integrators::explicit::vpl::IntegratorVPLOption::Volume,
                    _ => panic!("Invalid options: [all, surface, volume]"),
                }
            };

            let max_depth = match_infinity(m.value_of("max").unwrap());
            let nb_vpl = value_t_or_exit!(m.value_of("nb_vpl"), usize);
            let clamping = value_t_or_exit!(m.value_of("clamping"), f32);
            let option_vpl = get_option("option_vpl");
            let option_lt = get_option("option_lt");

            IntegratorType::Primal(Box::new(
                rustlight::integrators::explicit::vpl::IntegratorVPL {
                    nb_vpl,
                    max_depth,
                    clamping_factor: if clamping <= 0.0 {
                        None
                    } else {
                        Some(clamping)
                    },
                    option_vpl,
                    option_lt,
                },
            ))
        }
        ("uncorrelated_plane_single", Some(m)) => {
            let nb_primitive = value_t_or_exit!(m.value_of("nb_primitive"), usize);
            let strategy = value_t_or_exit!(m.value_of("strategy"), String);
            let strategy = match strategy.as_ref() {
                "uv" => rustlight::integrators::explicit::plane_single::SinglePlaneStrategy::UV,
                "ut" => rustlight::integrators::explicit::plane_single::SinglePlaneStrategy::UT,
                "vt" => rustlight::integrators::explicit::plane_single::SinglePlaneStrategy::VT,
                "average" => {
                    rustlight::integrators::explicit::plane_single::SinglePlaneStrategy::Average
                }
                "discrete_mis" => {
                    rustlight::integrators::explicit::plane_single::SinglePlaneStrategy::DiscreteMIS
                }
                "ualpha" => rustlight::integrators::explicit::plane_single::SinglePlaneStrategy::UAlpha,
                "cmis" => rustlight::integrators::explicit::plane_single::SinglePlaneStrategy::ContinousMIS,
                _ => panic!(
                    "{} is not a correct strategy choice (uv, ut, vt, average, discrete_mis, valpha, cmis)",
                    strategy
                )
            };
            IntegratorType::Primal(Box::new(
                rustlight::integrators::explicit::uncorrelated_plane_single::IntegratorSinglePlaneUncorrelated {
                    nb_primitive,
                    strategy,
                },
            ))
        }
        ("plane_single", Some(m)) => {
            let nb_primitive = value_t_or_exit!(m.value_of("nb_primitive"), usize);
            let strategy = value_t_or_exit!(m.value_of("strategy"), String);
            let strategy = match strategy.as_ref() {
                "uv" => rustlight::integrators::explicit::plane_single::SinglePlaneStrategy::UV,
                "ut" => rustlight::integrators::explicit::plane_single::SinglePlaneStrategy::UT,
                "vt" => rustlight::integrators::explicit::plane_single::SinglePlaneStrategy::VT,
                "average" => {
                    rustlight::integrators::explicit::plane_single::SinglePlaneStrategy::Average
                }
                "discrete_mis" => {
                    rustlight::integrators::explicit::plane_single::SinglePlaneStrategy::DiscreteMIS
                }
                "ualpha" => rustlight::integrators::explicit::plane_single::SinglePlaneStrategy::UAlpha,
                "cmis" => rustlight::integrators::explicit::plane_single::SinglePlaneStrategy::ContinousMIS,
                _ => panic!(
                    "{} is not a correct strategy choice (uv, ut, vt, average, discrete_mis, valpha, cmis)",
                    strategy
                )
            };
            IntegratorType::Primal(Box::new(
                rustlight::integrators::explicit::plane_single::IntegratorSinglePlane {
                    nb_primitive,
                    strategy,
                },
            ))
        }
        ("vol_primitives", Some(m)) => {
            let max_depth = match_infinity(m.value_of("max").unwrap());
            let nb_primitive = value_t_or_exit!(m.value_of("nb_primitive"), usize);
            let primitives = value_t_or_exit!(m.value_of("primitives"), String);
            let primitives = match primitives.as_ref() {
                "bre" => rustlight::integrators::explicit::vol_primitives::VolPrimitivies::BRE,
                "beam" => rustlight::integrators::explicit::vol_primitives::VolPrimitivies::Beams,
                "plane" => rustlight::integrators::explicit::vol_primitives::VolPrimitivies::Planes,
                "vrl" => rustlight::integrators::explicit::vol_primitives::VolPrimitivies::VRL,
                _ => panic!(
                    "{} is not a correct primitive (bre, beam, plane, vrl)",
                    primitives
                ),
            };
            IntegratorType::Primal(Box::new(
                rustlight::integrators::explicit::vol_primitives::IntegratorVolPrimitives {
                    nb_primitive,
                    max_depth,
                    primitives,
                },
            ))
        }
        ("pssmlt", Some(m)) => {
            let min_depth = match_infinity(m.value_of("min").unwrap());
            let max_depth = match_infinity(m.value_of("max").unwrap());
            let large_prob = value_t_or_exit!(m.value_of("large_prob"), f32);
            let nb_samples_norm = value_t_or_exit!(m.value_of("nb_samples_norm"), usize);
            let strategy = value_t_or_exit!(m.value_of("strategy"), String);
            let strategy = match strategy.as_ref() {
                "all" => {
                    rustlight::integrators::explicit::path::IntegratorPathTracingStrategies::All
                }
                "bsdf" => {
                    rustlight::integrators::explicit::path::IntegratorPathTracingStrategies::BSDF
                }
                "emitter" => {
                    rustlight::integrators::explicit::path::IntegratorPathTracingStrategies::Emitter
                }
                _ => panic!("invalid strategy: {}", strategy),
            };
            assert!(large_prob > 0.0 && large_prob <= 1.0);
            IntegratorType::Primal(Box::new(
                rustlight::integrators::mcmc::pssmlt::IntegratorPSSMLT {
                    large_prob,
                    nb_samples_norm,
                    integrator: Box::new(
                        rustlight::integrators::explicit::path::IntegratorPathTracing {
                            min_depth,
                            max_depth,
                            strategy,
                            single_scattering: false,
                        },
                    ),
                },
            ))
        }
        ("smcmc", Some(m)) => {
            let min_depth = match_infinity(m.value_of("min").unwrap());
            let max_depth = match_infinity(m.value_of("max").unwrap());
            let large_prob = value_t_or_exit!(m.value_of("large_prob"), f32);
            let strategy = value_t_or_exit!(m.value_of("strategy"), String);
            let strategy = match strategy.as_ref() {
                "all" => {
                    rustlight::integrators::explicit::path::IntegratorPathTracingStrategies::All
                }
                "bsdf" => {
                    rustlight::integrators::explicit::path::IntegratorPathTracingStrategies::BSDF
                }
                "emitter" => {
                    rustlight::integrators::explicit::path::IntegratorPathTracingStrategies::Emitter
                }
                _ => panic!("invalid strategy: {}", strategy),
            };
            assert!(large_prob > 0.0 && large_prob <= 1.0);
            let recons = value_t_or_exit!(m.value_of("recons"), String);
            let recons = recons.split(":").into_iter().map(|v| v).collect::<Vec<_>>();
            let recons: Box<dyn rustlight::integrators::mcmc::smcmc::Reconstruction> = match &recons
                [..]
            {
                ["naive"] => Box::new(rustlight::integrators::mcmc::smcmc::ReconstructionNaive {}),
                ["irls"] => Box::new(rustlight::integrators::mcmc::smcmc::ReconstructionIRLS {
                    irls_iter: 20,
                    internal_iter: 20,
                    alpha: 0.01,
                }),
                _ => panic!("invalid recons: {:?}", recons),
            };

            let init = value_t_or_exit!(m.value_of("init"), String);
            let init = init.split(":").into_iter().map(|v| v).collect::<Vec<_>>();
            let init: Box<dyn rustlight::integrators::mcmc::smcmc::Initialization> = match &init[..]
            {
                ["independent"] => {
                    Box::new(rustlight::integrators::mcmc::smcmc::IndependentInit { nb_spp: 16 })
                }
                ["mcmc"] => Box::new(rustlight::integrators::mcmc::smcmc::MCMCInit {
                    spp_mc: 1,
                    spp_mcmc: 8,
                    chain_length: 100,
                }),
                _ => panic!("invalid init: {:?}", init),
            };

            IntegratorType::Primal(Box::new(
                rustlight::integrators::mcmc::smcmc::StratifiedMCMC {
                    integrator: Box::new(
                        rustlight::integrators::explicit::path::IntegratorPathTracing {
                            min_depth,
                            max_depth,
                            strategy,
                            single_scattering: false,
                        },
                    ),
                    chains: None,
                    large_prob,
                    recons,
                    init,
                },
            ))
        }
        ("erpt", Some(m)) => {
            let min_depth = match_infinity(m.value_of("min").unwrap());
            let max_depth = match_infinity(m.value_of("max").unwrap());
            let strategy = value_t_or_exit!(m.value_of("strategy"), String);
            let strategy = match strategy.as_ref() {
                "all" => {
                    rustlight::integrators::explicit::path::IntegratorPathTracingStrategies::All
                }
                "bsdf" => {
                    rustlight::integrators::explicit::path::IntegratorPathTracingStrategies::BSDF
                }
                "emitter" => {
                    rustlight::integrators::explicit::path::IntegratorPathTracingStrategies::Emitter
                }
                _ => panic!("invalid strategy: {}", strategy),
            };
            let nb_mc = value_t_or_exit!(m.value_of("nb_mc"), usize);
            let chain_samples = value_t_or_exit!(m.value_of("chain_samples"), usize);
            let stratified = !m.is_present("no_stratified");
            IntegratorType::Primal(Box::new(
                rustlight::integrators::mcmc::erpt::IntegratorERPT {
                    nb_mc,
                    chain_samples,
                    integrator: Box::new(
                        rustlight::integrators::explicit::path::IntegratorPathTracing {
                            min_depth,
                            max_depth,
                            strategy,
                            single_scattering: false,
                        },
                    ),
                    stratified,
                },
            ))
        }
        ("ao", Some(m)) => {
            let normal_correction = m.is_present("normal-correction");
            let dist = match_infinity(m.value_of("distance").unwrap());
            IntegratorType::Primal(Box::new(rustlight::integrators::ao::IntegratorAO {
                max_distance: dist,
                normal_correction,
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

    // Read the sampler argument
    let sampler = value_t_or_exit!(matches.value_of("random_number_generator"), String);
    let sampler = sampler
        .split(":")
        .into_iter()
        .map(|v| v)
        .collect::<Vec<_>>();
    let mut sampler: Box<dyn rustlight::samplers::Sampler> = match &sampler[..] {
        ["independent"] => {
            Box::new(rustlight::samplers::independent::IndependentSampler::default())
        }
        ["independent", s] => Box::new(rustlight::samplers::independent::IndependentSampler {
            rnd: rand::rngs::SmallRng::seed_from_u64(
                s.parse::<u64>().expect("Seed need to be u64 type"),
            ),
        }),
        ["stratified"] => Box::new(rustlight::samplers::stratified::StratifiedSampler::create(
            nb_samples, 4,
        )),
        _ => panic!("Wrong sampler type provided {:?}", sampler),
    };

    let img = if matches.is_present("average") {
        let time_out = match_infinity(matches.value_of("average").unwrap());
        let mut int =
            IntegratorType::Primal(Box::new(rustlight::integrators::avg::IntegratorAverage {
                time_out,
                integrator: int,
            }));
        int.compute(sampler.as_mut(), &scene)
    } else {
        int.compute(sampler.as_mut(), &scene)
    };

    // Save the image
    img.save("primal", imgout_path_str);
}
