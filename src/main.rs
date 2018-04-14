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
// For print a progress bar
extern crate pbr;

use pbr::ProgressBar;
use cgmath::{Point2, Vector2};
use byteorder::{LittleEndian, WriteBytesExt};
use clap::{App, Arg, SubCommand};
use image::*;
use rustlight::integrator::ColorGradient;
use rustlight::Scale;
use rustlight::structure::Color;
use std::io::prelude::*;
use std::time::Instant;
use rayon::prelude::*;
use rustlight::sampler::SamplerMCMC;
use rustlight::sampler::Sampler;
use std::ops::AddAssign;
use rustlight::integrator::Integrator;
use rustlight::tools::StepRangeInt;
use std::sync::Mutex;

pub trait BitmapTrait: Default + AddAssign + Scale<f32> + Clone {}
impl BitmapTrait for ColorGradient {}
impl BitmapTrait for Color {}

/// Image block
/// for easy paralelisation over the thread
pub struct Bitmap<T: BitmapTrait> {
    pub pos: Point2<u32>,
    pub size: Vector2<u32>,
    pub pixels: Vec<T>,
}

impl<T: BitmapTrait> Bitmap<T> {
    pub fn new(pos: Point2<u32>, size: Vector2<u32>) -> Bitmap<T> {
        Bitmap {
            pos,
            size,
            pixels: vec![T::default(); (size.x * size.y) as usize],
        }
    }

    pub fn accumulate_bitmap(&mut self, o: &Bitmap<T>) {
        for x in 0..o.size.x {
            for y in 0..o.size.y {
                let c_p = Point2::new(o.pos.x + x, o.pos.y + y);
                self.accumulate(c_p, o.get(Point2::new(x, y)));
            }
        }
    }

    pub fn accumulate(&mut self, p: Point2<u32>, f: &T) {
        assert!(p.x < self.size.x);
        assert!(p.y < self.size.y);
        self.pixels[(p.y * self.size.y + p.x) as usize] += f.clone(); // FIXME: Not good for performance
    }

    pub fn accumulate_safe(&mut self, p: Point2<i32>, f: T) {
        if p.x >= 0 && p.y >= 0 && p.x < (self.size.x as i32) && p.y < (self.size.y as i32) {
            self.pixels[((p.y as u32) * self.size.y + p.x as u32) as usize] += f.clone(); // FIXME: Bad performance?
        }
    }

    pub fn get(&self, p: Point2<u32>) -> &T {
        assert!(p.x < self.size.x);
        assert!(p.y < self.size.y);
        &self.pixels[(p.y * self.size.y + p.x) as usize]
    }

    pub fn reset(&mut self) {
        self.pixels.iter_mut().for_each(|x| *x = T::default());
    }

    pub fn average(&self) -> T {
        let mut s = T::default();
        self.pixels.iter().for_each(|x| s += x.clone());
        s.scale(1.0 / self.pixels.len() as f32);
        s
    }
}

impl<T: BitmapTrait> Scale<f32> for Bitmap<T> {
    fn scale(&mut self, f: f32) {
        assert!(f > 0.0);
        self.pixels.iter_mut().for_each(|v| v.scale(f));
    }
}

impl<T: BitmapTrait> Iterator for Bitmap<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        unimplemented!()
    }
}

//////////////////////////////////
// Helpers
fn render<T: BitmapTrait + Send, I: Integrator<T> + Send + Sync>(
    scene: &::rustlight::scene::Scene,
    integrator: &I,
    nb_samples: usize,
) -> Bitmap<T> {
    assert_ne!(nb_samples, 0);

    // Create rendering blocks
    let mut image_blocks: Vec<Box<Bitmap<T>>> = Vec::new();
    for ix in StepRangeInt::new(0, scene.camera.size().x as usize, 16) {
        for iy in StepRangeInt::new(0, scene.camera.size().y as usize, 16) {
            let mut block = Bitmap::new(
                Point2 {
                    x: ix as u32,
                    y: iy as u32,
                },
                Vector2 {
                    x: std::cmp::min(16, scene.camera.size().x - ix as u32),
                    y: std::cmp::min(16, scene.camera.size().y - iy as u32),
                },
            );
            image_blocks.push(Box::new(block));
        }
    }

    // Render the image blocks
    let progress_bar = Mutex::new(ProgressBar::new(image_blocks.len() as u64));
    image_blocks.par_iter_mut().for_each(|im_block| {
        let mut sampler = rustlight::sampler::IndependentSampler::default();
        for ix in 0..im_block.size.x {
            for iy in 0..im_block.size.y {
                for _ in 0..nb_samples {
                    let c = integrator.compute(
                        (ix + im_block.pos.x, iy + im_block.pos.y),
                        scene,
                        &mut sampler,
                    );
                    im_block.accumulate(Point2 { x: ix, y: iy }, &c);
                }
            }
        }
        im_block.scale(1.0 / (nb_samples as f32));

        {
            progress_bar.lock().unwrap().inc();
        }
    });

    // Fill the image
    let mut image = Bitmap::new(Point2::new(0, 0), *scene.camera.size());
    for im_block in &image_blocks {
        image.accumulate_bitmap(im_block);
    }
    image
}

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

fn classical_mc_integration<T: Integrator<Color> + Send + Sync>(
    scene: &rustlight::scene::Scene,
    nb_samples: usize,
    nb_threads: Option<usize>,
    int: T,
) -> Bitmap<Color> {
    ////////////// Do the rendering
    info!("Rendering...");
    let start = Instant::now();
    let pool = match nb_threads {
        None => rayon::ThreadPoolBuilder::new(),
        Some(x) => rayon::ThreadPoolBuilder::new().num_threads(x),
    }.build()
        .unwrap();
    let img = pool.install(|| render(scene, &int, nb_samples));
    let elapsed = start.elapsed();
    info!(
        "Elapsed: {} ms",
        (elapsed.as_secs() * 1_000) + (elapsed.subsec_nanos() / 1_000_000) as u64
    );

    return img;
}

/// Compute the scene average luminance
/// Usefull for computing the normalisation factor for MCMC
fn integrate_image_plane<T: Integrator<Color> + Sync + Send>(
    scene: &rustlight::scene::Scene,
    integrator: &T,
    nb_samples: usize,
) -> f32 {
    assert_ne!(nb_samples, 0);

    let mut sampler = ::rustlight::sampler::IndependentSampler::default();
    (0..nb_samples)
        .into_iter()
        .map(|_i| {
            let x = (sampler.next() * scene.camera.size().x as f32) as u32;
            let y = (sampler.next() * scene.camera.size().y as f32) as u32;
            let c = integrator.compute((x, y), scene, &mut sampler);
            (c.r + c.g + c.b) / 3.0
        })
        .sum::<f32>() / (nb_samples as f32)
}

struct MCMCState {
    pub value: Color,
    pub tf: f32,
    pub pix: Point2<u32>,
    pub weight: f32,
}

impl MCMCState {
    pub fn new(v: Color, pix: Point2<u32>) -> MCMCState {
        MCMCState {
            value: v,
            tf: (v.r + v.g + v.b) / 3.0,
            pix,
            weight: 0.0,
        }
    }

    pub fn color(&self) -> Color {
        self.value * (self.weight / self.tf)
    }
}

fn classical_mcmc_integration<T: Integrator<Color> + Sync + Send>(
    scene: &rustlight::scene::Scene,
    nb_samples: usize,
    nb_threads: Option<usize>,
    int: T,
) -> Bitmap<Color> {
    ///////////// Prepare the pool for multiple thread
    let pool = match nb_threads {
        None => rayon::ThreadPoolBuilder::new(),
        Some(x) => rayon::ThreadPoolBuilder::new().num_threads(x),
    }.build()
        .unwrap();

    ///////////// Define the closure
    let sample = |s: &mut rustlight::sampler::IndependentSamplerReplay| {
        let x = (s.next() * scene.camera.size().x as f32) as u32;
        let y = (s.next() * scene.camera.size().y as f32) as u32;
        let c = { int.compute((x, y), scene, s) };
        MCMCState::new(c, Point2::new(x, y))
    };

    ///////////// Compute the normalization factor
    info!("Computing normalization factor...");
    let b = integrate_image_plane(scene, &int, 10000);
    info!("Normalisation factor: {:?}", b);

    ///////////// Compute the state initialization
    let nb_samples_total = nb_samples * (scene.camera.size().x * scene.camera.size().y) as usize;
    let nb_samples_per_chains = 100000;
    let nb_chains = nb_samples_total / nb_samples_per_chains;
    info!("Number of states: {:?}", nb_chains);
    // - Initialize the samplers
    let mut samplers = Vec::new();
    for _ in 0..nb_chains {
        samplers.push(rustlight::sampler::IndependentSamplerReplay::default());
    }

    ///////////// Compute the rendering (with the number of samples)
    info!("Rendering...");
    let start = Instant::now();
    let progress_bar = Mutex::new(ProgressBar::new(samplers.len() as u64));
    let img = Mutex::new(Bitmap::new(Point2::new(0, 0), *scene.camera.size()));
    pool.install(|| {
        samplers.par_iter_mut().for_each(|s| {
            // Initialize the sampler
            s.large_step = true;
            let mut current_state = sample(s);
            while current_state.tf == 0.0 {
                s.reject();
                current_state = sample(s);
            }
            s.accept();

            let mut my_img: Bitmap<Color> = Bitmap::new(Point2::new(0, 0), *scene.camera.size());
            (0..nb_samples_per_chains).into_iter().for_each(|_| {
                // Choose randomly between large and small perturbation
                s.large_step = s.rand() < 0.3;
                let mut proposed_state = sample(s);
                let accept_prob = (proposed_state.tf / current_state.tf).min(1.0);
                // Do waste reclycling
                current_state.weight += 1.0 - accept_prob;
                proposed_state.weight += accept_prob;
                if accept_prob > s.rand() {
                    my_img.accumulate(current_state.pix, &current_state.color());
                    s.accept();
                    current_state = proposed_state;
                } else {
                    my_img.accumulate(proposed_state.pix, &proposed_state.color());
                    s.reject();
                }
            });
            // Flush the last state
            my_img.accumulate(current_state.pix, &current_state.color());

            my_img.scale(1.0 / (nb_samples_per_chains as f32));
            {
                img.lock().unwrap().accumulate_bitmap(&my_img);
                progress_bar.lock().unwrap().inc();
            }
        });
    });
    let mut img: Bitmap<Color> = img.into_inner().unwrap();
    let elapsed = start.elapsed();
    info!(
        "Elapsed: {} ms",
        (elapsed.as_secs() * 1_000) + (elapsed.subsec_nanos() / 1_000_000) as u64
    );

    // ==== Compute and scale to the normalization factor
    let img_avg = img.average();
    let img_avg_lum = (img_avg.r + img_avg.g + img_avg.b) / 3.0;
    img.scale(b / img_avg_lum);

    img
}

fn gradient_domain_integration<T: Integrator<ColorGradient> + Sync + Send>(
    scene: &rustlight::scene::Scene,
    nb_samples: usize,
    nb_threads: Option<usize>,
    int: T,
) -> Bitmap<Color> {
    info!("Rendering...");
    let start = Instant::now();
    let pool = match nb_threads {
        None => rayon::ThreadPoolBuilder::new(),
        Some(x) => rayon::ThreadPoolBuilder::new().num_threads(x),
    }.build()
        .unwrap();
    let img_grad = pool.install(|| render(scene, &int, nb_samples));
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
            SubCommand::with_name("pssmlt")
                .about("path tracing with MCMC sampling")
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
    let scene = rustlight::scene::Scene::new(&data, wk).expect("error when loading the scene");

    let img = match matches.subcommand() {
        ("path-explicit", Some(m)) => {
            let max_depth = match_infinity(m.value_of("max").unwrap());

            classical_mc_integration(
                &scene,
                nb_samples,
                nb_threads,
                rustlight::integrator::IntegratorUniPath { max_depth },
            )
        }
        ("path", Some(m)) => {
            let max_depth = match_infinity(m.value_of("max").unwrap());
            let min_depth = match_infinity(m.value_of("min").unwrap());

            classical_mc_integration(
                &scene,
                nb_samples,
                nb_threads,
                rustlight::integrator::IntegratorPath {
                    max_depth,
                    min_depth,
                },
            )
        }
        ("pssmlt", Some(m)) => {
            let max_depth = match_infinity(m.value_of("max").unwrap());
            let min_depth = match_infinity(m.value_of("min").unwrap());

            classical_mcmc_integration(
                &scene,
                nb_samples,
                nb_threads,
                rustlight::integrator::IntegratorPath {
                    max_depth,
                    min_depth,
                },
            )
        }
        ("gd-path", Some(m)) => {
            let max_depth = match_infinity(m.value_of("max").unwrap());
            let min_depth = match_infinity(m.value_of("min").unwrap());

            gradient_domain_integration(
                &scene,
                nb_samples,
                nb_threads,
                rustlight::integrator::IntegratorPath {
                    max_depth,
                    min_depth,
                },
            )
        }
        ("ao", Some(m)) => {
            let dist = match_infinity(m.value_of("distance").unwrap());

            classical_mc_integration(
                &scene,
                nb_samples,
                nb_threads,
                rustlight::integrator::IntegratorAO { max_distance: dist },
            )
        }
        ("direct", Some(m)) => classical_mc_integration(
            &scene,
            nb_samples,
            nb_threads,
            rustlight::integrator::IntegratorDirect {
                nb_bsdf_samples: value_t_or_exit!(m.value_of("bsdf"), u32),
                nb_light_samples: value_t_or_exit!(m.value_of("light"), u32),
            },
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
