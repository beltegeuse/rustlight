#![cfg_attr(feature = "clippy", feature(plugin))]
#![cfg_attr(feature = "clippy", plugin(clippy))]
#![allow(dead_code)]
#![allow(clippy::float_cmp)]
#![allow(clippy::cognitive_complexity)]

extern crate cgmath;
extern crate clap;
extern crate log4rs;
extern crate num_cpus;
#[macro_use]
extern crate log;
extern crate rand;
extern crate rayon;
extern crate rustlight;

use clap::{Args, Parser, Subcommand, ValueEnum};

use log::LevelFilter;
use log4rs::{
    append::{
        console::{ConsoleAppender, Target},
        file::FileAppender,
    },
    config::{Appender, Config, Root},
    encode::pattern::PatternEncoder,
    filter::threshold::ThresholdFilter,
};
use rand::SeedableRng;
use rustlight::integrators::IntegratorType;
fn match_infinity<T: std::str::FromStr>(input: String) -> Option<T> {
    match input.as_str() {
        "inf" => None,
        _ => match input.parse::<T>() {
            Ok(x) => Some(x),
            Err(_e) => panic!("wrong input for inf type parameter"),
        },
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum ExtraOptions {
    /// Use adaptive tree splitting
    ATS,
    /// Remove shading normals
    NoShading,
    /// HVS Lights (hard coded)
    HVSLight,
    /// Texture Lights
    TextureLight,
}

#[derive(Debug, Args)]
pub struct PathLength {
    #[arg(long, short = 'm', default_value = "inf")]
    max_depth: String,
    #[arg(long, short = 'n', default_value = "0")]
    min_depth: String,
    #[arg(long, short = 'r', default_value = "0")]
    rr_depth: String,
}
impl PathLength {
    pub fn parse<T: std::str::FromStr>(self) -> (Option<T>, Option<T>, Option<T>) {
        (
            match_infinity(self.min_depth),
            match_infinity(self.max_depth),
            match_infinity(self.rr_depth),
        )
    }
}

#[derive(Debug, Args)]
pub struct Reconstruction {
    #[arg(long, short = 'i', default_value_t = 50)]
    iterations: usize,
    #[arg(long, default_value = "uniform")]
    strategy: String,
}
impl Reconstruction {
    pub fn parse(
        self,
        nb_samples: usize,
    ) -> Box<dyn rustlight::integrators::PoissonReconstruction + Sync> {
        match self.strategy.as_str() {
            "uniform" => Box::new(
                rustlight::integrators::gradient::recons::UniformPoissonReconstruction {
                    iterations: self.iterations,
                },
            ),
            "weighted" => Box::new(
                rustlight::integrators::gradient::recons::WeightedPoissonReconstruction::new(
                    self.iterations,
                ),
            ),
            "bagging" => Box::new(
                rustlight::integrators::gradient::recons::BaggingPoissonReconstruction {
                    iterations: self.iterations,
                    nb_buffers: if nb_samples <= 8 { nb_samples } else { 8 },
                },
            ),
            _ => panic!("Impossible to found a reconstruction_type"),
        }
    }
}

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Scene description file
    #[arg(value_name = "FILE", default_value = "test.pfm")]
    scene: String,
    /// Number of samples
    #[arg(long, short, default_value_t = 1)]
    nbsamples: usize,
    /// Number averaging pass ('inf' possible)
    #[arg(long, short)]
    average: Option<String>,
    /// Number of threads
    #[arg(long, short, allow_hyphen_values(true))]
    threads: Option<i32>,
    /// Random number generator
    #[arg(long, short, default_value = "independent")]
    random_number_generator: String,
    /// Image scale (to faster or slower rendering)
    #[arg(long, short, default_value_t = 1_f32)]
    scale_image: f32,
    /// Equal time comparison
    #[arg(long, short)]
    equal_time: Option<f32>, // False as default?
    /// Output image file
    #[arg(long, short, value_name = "FILE")]
    output: String,
    /// Infinite medium with density
    #[arg(long, short, default_value = "0.0")]
    medium: String,
    /// Logs
    #[arg(long, short)]
    log: Option<String>,
    /// Options
    #[arg(long, short)]
    xtra_options: Vec<ExtraOptions>,

    #[clap(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    // Regular rendering algorithms
    AO {
        #[arg(long, short, default_value = "1.0")]
        distance: String,
        #[arg(long, short)]
        normal_correction: bool,
    },
    Direct {
        #[arg(long, short = 'b', default_value_t = 1)]
        nb_bsdf_samples: usize,
        #[arg(long, short = 'l', default_value_t = 1)]
        nb_light_samples: usize,
    },
    Path {
        #[command(flatten)]
        path_length: PathLength,
        #[arg(long, short = 'x')]
        single_scattering: bool,
        #[arg(long, short, default_value = "all")]
        strategy: String,
    },
    LightTracing {
        #[command(flatten)]
        path_length: PathLength,
        #[arg(long, short, default_value = "all")]
        strategy: String,
    },
    VPL {
        #[command(flatten)]
        path_length: PathLength,
        #[arg(long, short = 'b', default_value_t = 0.0)]
        clamping: f32,
        #[arg(long, short, default_value_t = 128)]
        nb_vpl: usize,
        #[arg(long, short = 'l', default_value = "all")]
        option_lt: String,
        #[arg(long, short = 'v', default_value = "all")]
        option_vpl: String,
    },
    // Volume
    VolPrimitivies {
        #[command(flatten)]
        path_length: PathLength,
        #[arg(long, short, default_value_t = 128)]
        nb_primitive: usize,
        #[arg(long, short, default_value = "BRE")]
        primitives: String,
    },
    PlaneSingle {
        #[arg(long, short, default_value_t = 128)]
        nb_primitive: usize,
        #[arg(long, short, default_value = "average")]
        strategy: String,
    },
    UncorrelatedPlaneSingle {
        #[arg(long, short, default_value_t = 128)]
        nb_primitive: usize,
        #[arg(long, short, default_value = "average")]
        strategy: String,
    },
    PointNormal {
        #[arg(long, short = 'z')]
        disable_aa: bool,
        #[arg(long, short = 'k')]
        splitting: Option<f32>,
        #[arg(long, short = 'x')]
        use_mis: bool,
        /// warps: P = Phase, T = Tr, N = PN
        #[arg(long, short = 'w')]
        warps: String,
        /// warps_type: L = Linear, B = Bezier
        #[arg(long, short = 'w', default_value = "L")]
        warps_strategy: String,
        #[arg(long, short, default_value = "tr_ex")]
        strategy: String,
    },
    // Gradient
    GradientPath {
        #[command(flatten)]
        path_length: PathLength,
        #[command(flatten)]
        recons: Reconstruction,
    },
    GradientPathExplicit {
        #[command(flatten)]
        path_length: PathLength,
        #[command(flatten)]
        recons: Reconstruction,
        #[arg(long, short = 's', default_value_t = 1.0)]
        min_survival: f32,
    },
    // MCMC
    PSSMLT {
        #[command(flatten)]
        path_length: PathLength,
        #[arg(long, short, default_value = "all")]
        strategy: String,
        #[arg(long, short, default_value_t = 0.3)]
        large_prob: f32,
        #[arg(long, short = 'b', default_value_t = 100000)]
        nb_samples_norm: usize,
    },
    SMCMC {
        #[command(flatten)]
        path_length: PathLength,
        #[arg(long, short, default_value = "all")]
        strategy: String,
        #[arg(long, short, default_value_t = 0.3)]
        large_prob: f32,
        #[arg(long, short, default_value = "irls", short = 'p')]
        recons: String,
        #[arg(long, short, default_value = "mcmc")]
        init: String,
    },
    ERPT {
        #[command(flatten)]
        path_length: PathLength,
        #[arg(long, short = 'k')]
        stratified: bool,
        #[arg(long, short, default_value = "all")]
        strategy: String,
        #[arg(long, short = 'e', default_value_t = 1)]
        nb_mc: usize,
        #[arg(long, short, default_value_t = 100)]
        chain_samples: usize,
    },
}

fn main() {
    let cli = Cli::parse();

    /////////////// Setup logging system
    let _handle = {
        let level = log::LevelFilter::Info;
        let stderr = ConsoleAppender::builder()
            .target(Target::Stderr)
            .encoder(Box::new(PatternEncoder::new("{l} {M} - {m}\n")))
            .build();
        let config = if let Some(log) = cli.log {
            let logfile = FileAppender::builder()
                // Pattern: https://docs.rs/log4rs/*/log4rs/encode/pattern/index.html
                .encoder(Box::new(PatternEncoder::new("{l} {M} - {m}\n")))
                .build(log)
                .unwrap();

            Config::builder()
                .appender(Appender::builder().build("logfile", Box::new(logfile)))
                .appender(
                    Appender::builder()
                        .filter(Box::new(ThresholdFilter::new(level)))
                        .build("stderr", Box::new(stderr)),
                )
                .build(
                    Root::builder()
                        .appender("logfile")
                        .appender("stderr")
                        .build(level),
                )
                .unwrap()
        } else {
            Config::builder()
                .appender(
                    Appender::builder()
                        .filter(Box::new(ThresholdFilter::new(level)))
                        .build("stderr", Box::new(stderr)),
                )
                .build(Root::builder().appender("stderr").build(LevelFilter::Trace))
                .unwrap()
        };

        log4rs::init_config(config).unwrap()
    };

    //////////////// Load the rendering configuration
    let options = cli.xtra_options;
    info!("Extra options: {:?}", options);
    let use_ats = options.contains(&ExtraOptions::ATS);
    let remove_shading_normals = options.contains(&ExtraOptions::NoShading);
    let hsv_lights = options.contains(&ExtraOptions::HVSLight);
    let texture_lights = options.contains(&ExtraOptions::TextureLight);

    //////////////// Load the scene

    let scene = rustlight::scene_loader::SceneLoaderManager::default()
        .load(cli.scene.to_string(), !remove_shading_normals)
        .expect("error on loading the scene");
    let scene = match cli.threads {
        None => scene,
        Some(v) => match v {
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
        },
    };
    let mut scene = scene.nb_samples(cli.nbsamples).output_img(&cli.output);

    ///////////////// Medium
    {
        let medium_density = cli.medium;
        let medium_density = medium_density
            .split(":")
            .into_iter()
            .map(|v| v)
            .collect::<Vec<_>>();
        let (sigma_s, sigma_a, phase) = match &medium_density[..] {
            [sigma_s] => (
                sigma_s.parse().unwrap(),
                0.0,
                rustlight::volume::PhaseFunction::Isotropic(),
            ),
            [sigma_s, sigma_a] => (
                sigma_s.parse().unwrap(),
                sigma_a.parse().unwrap(),
                rustlight::volume::PhaseFunction::Isotropic(),
            ),
            [sigma_s, sigma_a, g] => (
                sigma_s.parse().unwrap(),
                sigma_a.parse().unwrap(),
                rustlight::volume::PhaseFunction::HenyeyGreenstein(g.parse::<f32>().unwrap()),
            ),

            _ => panic!("invalid medium_density: {:?}", medium_density),
        };

        if sigma_a + sigma_s != 0.0 {
            let density_mult = 1.0; //*matches.get_one::<f32>("density_mult").expect("density_mult");
            let sigma_a = rustlight::structure::Color::value(sigma_a);
            let sigma_s = rustlight::structure::Color::value(sigma_s);
            let sigma_t = (sigma_a + sigma_s) * density_mult;
            scene.volume = Some(rustlight::volume::HomogenousVolume {
                sigma_a,
                sigma_s,
                sigma_t,
                phase,
            });

            info!("Create volume with: ");
            info!(" - sigma_a: {:?}", sigma_a);
            info!(" - sigma_s: {:?}", sigma_s);
            info!(" - sigma_t: {:?}", sigma_t);
        }
    }
    ///////////////// Tweak the image size
    {
        let image_scale = cli.scale_image;
        if image_scale != 1.0 {
            info!("Scale the image: {:?}", image_scale);
            assert!(image_scale != 0.0);
            scene.camera.scale_image(image_scale);
        }
    }
    // ///////////////// Overide light is needed
    if hsv_lights || texture_lights {
        for m in &mut scene.meshes {
            if m.is_light() {
                let scale = match m.emission {
                    rustlight::geometry::EmissionType::Color { v } => v.luminance(),
                    _ => 1.0,
                };

                let m = std::sync::Arc::get_mut(m).unwrap();
                if hsv_lights {
                    m.emission = rustlight::geometry::EmissionType::HSV { scale };
                } else {
                    m.emission = rustlight::geometry::EmissionType::Texture {
                        scale,
                        img: rustlight::structure::Bitmap::read("butterfly.jpg"),
                    };
                }
            }
        }
    }

    // Build internal
    scene.build_emitters(use_ats);

    // ///////////////// Create the main integrator
    let mut int = match cli.command {
        Commands::PointNormal {
            disable_aa,
            splitting,
            use_mis,
            warps,
            warps_strategy,
            strategy,
        } => {
            use rustlight::integrators::explicit::point_normal::Strategies;
            use rustlight::integrators::explicit::point_normal::WrapStrategy;
            // let order = m.get_one::<>("order"), usize);
            let warps_strategy = match warps_strategy.as_ref() {
                "L" => WrapStrategy::Linear,
                "B" => WrapStrategy::Bezier,
                _ => panic!("Need to choose between L and B"),
            };
            info!("Strategy: {}", strategy);
            let strategy = match strategy.as_ref() {
                // Equi-angular variants
                "eq_phase" => Strategies::EQUIANGULAR | Strategies::PHASE,
                "eq_ex" => Strategies::EQUIANGULAR | Strategies::EX,
                "eq_best_ex" => Strategies::EQUIANGULAR | Strategies::EX | Strategies::BEST,
                "eq_warp_ex" => Strategies::EQUIANGULAR | Strategies::EX | Strategies::WRAP,
                "eq_phase_taylor_ex" => {
                    Strategies::EQUIANGULAR | Strategies::TAYLOR_PHASE | Strategies::EX
                }
                "eq_tr_taylor_ex" => {
                    Strategies::EQUIANGULAR | Strategies::TAYLOR_TR | Strategies::EX
                }
                // Equi-angular with clamped angles
                "eq_clamped_phase" => Strategies::EQUIANGULAR_CLAMPED | Strategies::PHASE, //< Biased
                "eq_clamped_ex" => Strategies::EQUIANGULAR_CLAMPED | Strategies::EX,
                "eq_clamped_best_ex" => {
                    Strategies::EQUIANGULAR_CLAMPED | Strategies::EX | Strategies::BEST
                }
                "eq_clamped_warp_ex" => {
                    Strategies::EQUIANGULAR_CLAMPED | Strategies::EX | Strategies::WRAP
                }
                "eq_clamped_phase_taylor_ex" => {
                    Strategies::EQUIANGULAR_CLAMPED | Strategies::EX | Strategies::TAYLOR_PHASE
                }
                "eq_clamped_tr_taylor_ex" => {
                    Strategies::EQUIANGULAR_CLAMPED | Strategies::TAYLOR_TR | Strategies::EX
                }
                // TR
                "tr_ex" => Strategies::TR | Strategies::EX,
                "tr_phase" => Strategies::TR | Strategies::PHASE,
                // PN
                "pn_ex" => Strategies::POINT_NORMAL | Strategies::EX,
                "pn_tr_taylor_ex" => {
                    Strategies::POINT_NORMAL | Strategies::TAYLOR_TR | Strategies::EX
                }
                "pn_phase_taylor_ex" => {
                    Strategies::POINT_NORMAL | Strategies::TAYLOR_PHASE | Strategies::EX
                }
                "pn_warp_ex" => Strategies::POINT_NORMAL | Strategies::WRAP | Strategies::EX,
                "pn_best_ex" => Strategies::POINT_NORMAL | Strategies::EX | Strategies::BEST,
                _ => panic!("invalid strategy: {}", strategy),
            };

            info!(" - use_mis       : {}", use_mis);
            info!(" - warps         : {}", warps);
            info!(" - warps_strategy: {:?}", warps_strategy);
            info!(" - splitting         : {:?}", splitting);

            info!(" - use_mis       : {}", use_mis);
            info!(" - warps         : {}", warps);
            info!(" - warps_strategy: {:?}", warps_strategy);
            info!(" - splitting         : {:?}", splitting);

            info!(" - use_mis       : {}", use_mis);
            info!(" - warps         : {}", warps);
            info!(" - warps_strategy: {:?}", warps_strategy);
            info!(" - splitting         : {:?}", splitting);

            IntegratorType::Primal(Box::new(
                rustlight::integrators::explicit::point_normal::IntegratorPointNormal {
                    strategy,
                    use_mis,
                    warps: warps.to_string(),
                    warps_strategy,
                    splitting,
                    use_aa: !disable_aa,
                },
            ))
        }
        Commands::Path {
            path_length,
            single_scattering,
            strategy,
        } => {
            let (min_depth, max_depth, rr_depth) = path_length.parse();
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
                    rr_depth,
                    strategy,
                    single_scattering,
                },
            ))
        }
        Commands::LightTracing {
            path_length,
            strategy,
        } => {
            let (min_depth, max_depth, rr_depth) = path_length.parse();
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
                    rr_depth,
                    render_surface,
                    render_volume,
                },
            ))
        }
        Commands::GradientPath {
            path_length,
            recons,
        } => {
            let (min_depth, max_depth, _rr_depth) = path_length.parse();
            let recons = recons.parse(cli.nbsamples);
            IntegratorType::Gradient(Box::new(
                rustlight::integrators::gradient::path::IntegratorGradientPath {
                    max_depth,
                    min_depth,
                    recons,
                },
            ))
        }
        Commands::GradientPathExplicit {
            path_length,
            recons,
            min_survival,
        } => {
            let (_min_depth, max_depth, _rr_depth) = path_length.parse();
            if min_survival <= 0.0 || min_survival > 1.0 {
                panic!("need to specify min_survival in ]0.0,1.0]");
            }
            let recons = recons.parse(cli.nbsamples);
            IntegratorType::Gradient(Box::new(
                rustlight::integrators::gradient::explicit::IntegratorGradientPathTracing {
                    max_depth,
                    recons: recons,
                    min_survival: Some(min_survival),
                },
            ))
        }
        Commands::VPL {
            path_length,
            clamping,
            nb_vpl,
            option_lt,
            option_vpl,
        } => {
            let get_option = |value: String| match value.as_str() {
                "all" => rustlight::integrators::explicit::vpl::IntegratorVPLOption::All,
                "surface" => rustlight::integrators::explicit::vpl::IntegratorVPLOption::Surface,
                "volume" => rustlight::integrators::explicit::vpl::IntegratorVPLOption::Volume,
                _ => panic!("Invalid options: [all, surface, volume]"),
            };
            let (_min_depth, max_depth, rr_depth) = path_length.parse();

            let option_vpl = get_option(option_vpl);
            let option_lt = get_option(option_lt);

            IntegratorType::Primal(Box::new(
                rustlight::integrators::explicit::vpl::IntegratorVPL {
                    nb_vpl,
                    max_depth,
                    rr_depth,
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
        Commands::UncorrelatedPlaneSingle {
            nb_primitive,
            strategy,
        } => {
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
        Commands::PlaneSingle {
            nb_primitive,
            strategy,
        } => {
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
        Commands::VolPrimitivies {
            path_length,
            nb_primitive,
            primitives,
        } => {
            let (_min_depth, max_depth, rr_depth) = path_length.parse();
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
                    rr_depth,
                    primitives,
                },
            ))
        }
        Commands::PSSMLT {
            path_length,
            strategy,
            large_prob,
            nb_samples_norm,
        } => {
            let (min_depth, max_depth, rr_depth) = path_length.parse();
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
                            rr_depth,
                            strategy,
                            single_scattering: false,
                        },
                    ),
                },
            ))
        }
        Commands::SMCMC {
            path_length,
            strategy,
            large_prob,
            recons,
            init,
        } => {
            let (min_depth, max_depth, _rr_depth) = path_length.parse();
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
                            rr_depth: None, // Disable RR for now
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
        Commands::ERPT {
            path_length,
            stratified,
            strategy,
            nb_mc,
            chain_samples,
        } => {
            let (min_depth, max_depth, rr_depth) = path_length.parse();
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
                rustlight::integrators::mcmc::erpt::IntegratorERPT {
                    nb_mc,
                    chain_samples,
                    integrator: Box::new(
                        rustlight::integrators::explicit::path::IntegratorPathTracing {
                            min_depth,
                            max_depth,
                            rr_depth,
                            strategy,
                            single_scattering: false,
                        },
                    ),
                    stratified,
                },
            ))
        }
        Commands::AO {
            distance,
            normal_correction,
        } => {
            let max_distance = match_infinity(distance);
            IntegratorType::Primal(Box::new(rustlight::integrators::ao::IntegratorAO {
                max_distance,
                normal_correction,
            }))
        }
        Commands::Direct {
            nb_bsdf_samples,
            nb_light_samples,
        } => IntegratorType::Primal(Box::new(rustlight::integrators::direct::IntegratorDirect {
            nb_bsdf_samples,
            nb_light_samples,
        })),
    };

    // Read the sampler argument
    let sampler = cli
        .random_number_generator
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
            cli.nbsamples,
            4,
        )),
        _ => panic!("Wrong sampler type provided {:?}", sampler),
    };

    let img = if let Some(time_out) = cli.equal_time {
        let time_out = time_out * 1000.0;
        info!("Time out in ms: {}", time_out);
        let mut int = IntegratorType::Primal(Box::new(
            rustlight::integrators::equal_time::IntegratorEqualTime {
                target_time_ms: time_out as u128,
                integrator: int,
            },
        ));
        int.compute(sampler.as_mut(), &scene)
    } else if let Some(v) = cli.average {
        let time_out = match_infinity(v);
        let mut int =
            IntegratorType::Primal(Box::new(rustlight::integrators::avg::IntegratorAverage {
                time_out,
                integrator: int,
                dump_all: true,
            }));
        int.compute(sampler.as_mut(), &scene)
    } else {
        int.compute(sampler.as_mut(), &scene)
    };

    // Save the image
    info!("Save final image: {}", cli.output);
    img.save("primal", &cli.output);
}
