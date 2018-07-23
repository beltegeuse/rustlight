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

use byteorder::{LittleEndian, WriteBytesExt};
use cgmath::{Point2, Vector2};
use clap::{App, Arg, SubCommand};
use image::*;
use pbr::ProgressBar;
use rayon::prelude::*;
use rustlight::integrators::gradient_domain::*;
use rustlight::integrators::*;
use rustlight::samplers::Sampler;
use rustlight::samplers::SamplerMCMC;
use rustlight::structure::Color;
use rustlight::tools::StepRangeInt;
use rustlight::Scale;
use std::io::prelude::*;
use std::ops::AddAssign;
use std::sync::Mutex;
use std::time::Instant;

/// The different way to store the informations
pub trait BitmapTrait: Default + AddAssign + Scale<f32> + Clone {
    // Number of channel needed for the bitmap storage
    fn size() -> usize;
    fn get(&self, channel: usize) -> Color;
    fn index(name: &str) -> usize;
}
impl BitmapTrait for ColorGradient {
    fn size() -> usize {
        1 + 4 + 1 // Very direct + Throughput + Gradient
    }
    fn get(&self, channel: usize) -> Color {
        match channel {
            0 => self.very_direct.clone(),
            1 => self.main.clone(),
            2 => self.gradients[0].clone(),
            3 => self.gradients[1].clone(),
            4 => self.gradients[2].clone(),
            5 => self.gradients[3].clone(),
            _ => unimplemented!(),
        }
    }
    fn index(name: &str) -> usize {
        match name {
            "very_direct" => 0,
            "main" => 1,
            "dy_pos" => 2,
            "dy_neg" => 3,
            "dx_pos" => 4,
            "dx_neg" => 5,
            _ => unimplemented!(),
        }
    }
}
impl BitmapTrait for Color {
    fn size() -> usize {
        1
    }
    fn get(&self, _channel: usize) -> Color {
        self.clone()
    }
    fn index(name: &str) -> usize {
        0
    }
}

/// Image block
/// for easy paralelisation over the thread
pub struct Bitmap {
    pub pos: Point2<u32>,
    pub size: Vector2<u32>,
    pub pixels: Vec<Vec<Color>>,
    pub variances: Vec<Vec<VarianceEstimator>>,
}

impl Bitmap {
    pub fn new(nb_channels: usize, pos: Point2<u32>, size: Vector2<u32>) -> Bitmap {
        let mut pixels = vec![];
        let mut variances = vec![];
        for _ in 0..nb_channels {
            pixels.push(vec![Color::default(); (size.x * size.y) as usize]);
            variances.push(vec![
                VarianceEstimator::default();
                (size.x * size.y) as usize
            ]);
        }
        Bitmap {
            pos,
            size,
            pixels,
            variances,
        }
    }

    pub fn accumulate_bitmap(&mut self, o: &Bitmap) {
        assert!(o.pixels.len() == self.pixels.len());
        // This is special, it does not allowed to write twice the same pixels
        // This function is only when we

        for channel in 0..self.pixels.len() {
            for y in 0..o.size.y {
                for x in 0..o.size.x {
                    let p = Point2::new(o.pos.x + x, o.pos.y + y);
                    let index = (p.y * self.size.y + p.x) as usize;
                    let index_other = (y * o.size.y + x) as usize;
                    self.pixels[channel][index] = o.pixels[channel][index_other];
                    self.variances[channel][index] = o.variances[channel][index_other];
                }
            }
        }
    }

    pub fn accumulate(&mut self, p: Point2<u32>, f: Color, channel: usize) {
        assert!(p.x < self.size.x);
        assert!(p.y < self.size.y);
        assert!(self.pixels.len() > channel);
        let index = (p.y * self.size.y + p.x) as usize;
        self.pixels[channel][index] += f;
        self.variances[channel][index].add(f.luminance());
    }

    pub fn accumulate_safe(&mut self, p: Point2<i32>, f: Color, channel: usize) {
        if p.x >= 0 && p.y >= 0 && p.x < (self.size.x as i32) && p.y < (self.size.y as i32) {
            self.accumulate(
                Point2 {
                    x: p.x as u32,
                    y: p.y as u32,
                },
                f,
                channel,
            );
        }
    }

    pub fn get(&self, p: Point2<u32>, channel: usize) -> &Color {
        assert!(p.x < self.size.x);
        assert!(p.y < self.size.y);
        &self.pixels[channel][(p.y * self.size.y + p.x) as usize]
    }

    pub fn get_variance(&self, p: Point2<u32>, channel: usize) -> f32 {
        assert!(p.x < self.size.x);
        assert!(p.y < self.size.y);
        self.variances[channel][(p.y * self.size.y + p.x) as usize].variance()
    }

    pub fn reset(&mut self) {
        for channel in 0..self.pixels.len() {
            self.pixels[channel]
                .iter_mut()
                .for_each(|x| *x = Color::default());
            self.variances[channel]
                .iter_mut()
                .for_each(|x| *x = VarianceEstimator::default());
        }
    }

    pub fn average_pixel(&self, channel: usize) -> Color {
        let mut s = Color::default();
        self.pixels[channel].iter().for_each(|x| s += x.clone());
        s.scale(1.0 / self.pixels.len() as f32);
        s
    }
}

impl Scale<f32> for Bitmap {
    fn scale(&mut self, f: f32) {
        assert!(f > 0.0);
        for channel in 0..self.pixels.len() {
            self.pixels[channel].iter_mut().for_each(|v| v.scale(f));
        }
    }
}

impl Iterator for Bitmap {
    type Item = Color;

    fn next(&mut self) -> Option<Self::Item> {
        unimplemented!()
    }
}

#[derive(Clone, Debug, Copy)]
pub struct VarianceEstimator {
    pub mean: f32,
    pub mean_sqr: f32,
    pub sample_count: u32,
}
impl VarianceEstimator {
    fn add(&mut self, v: f32) {
        self.sample_count += 1;
        let delta = v - self.mean;
        self.mean += delta / self.sample_count as f32;
        self.mean_sqr += delta * (v - self.mean);
    }

    fn variance(&self) -> f32 {
        self.mean_sqr / (self.sample_count - 1) as f32
    }
}
impl Default for VarianceEstimator {
    fn default() -> Self {
        Self {
            mean: 0.0,
            mean_sqr: 0.0,
            sample_count: 0,
        }
    }
}

//////////////////////////////////
// Helpers
fn render<T: BitmapTrait + Send, I: Integrator<T>>(
    scene: &::rustlight::scene::Scene,
    integrator: &I,
    nb_samples: usize,
) -> Bitmap {
    assert_ne!(nb_samples, 0);

    // Create rendering blocks
    let mut image_blocks: Vec<Box<Bitmap>> = Vec::new();
    for ix in StepRangeInt::new(0, scene.camera.size().x as usize, 16) {
        for iy in StepRangeInt::new(0, scene.camera.size().y as usize, 16) {
            let mut block = Bitmap::new(
                T::size(),
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
        let mut sampler = rustlight::samplers::independent::IndependentSampler::default();
        for ix in 0..im_block.size.x {
            for iy in 0..im_block.size.y {
                for _ in 0..nb_samples {
                    let c = integrator.compute(
                        (ix + im_block.pos.x, iy + im_block.pos.y),
                        scene,
                        &mut sampler,
                    );
                    for channel in 0..T::size() {
                        im_block.accumulate(Point2 { x: ix, y: iy }, c.get(channel), channel);
                    }
                }
            }
        }
        im_block.scale(1.0 / (nb_samples as f32));

        {
            progress_bar.lock().unwrap().inc();
        }
    });

    // Fill the image
    let mut image = Bitmap::new(T::size(), Point2::new(0, 0), *scene.camera.size());
    for im_block in &image_blocks {
        image.accumulate_bitmap(im_block);
    }
    image
}

fn save_pfm(imgout_path_str: &str, img: &Bitmap) {
    let mut file = std::fs::File::create(std::path::Path::new(imgout_path_str)).unwrap();
    let header = format!("PF\n{} {}\n-1.0\n", img.size.y, img.size.x);
    file.write(header.as_bytes()).unwrap();
    for y in 0..img.size.y {
        for x in 0..img.size.x {
            let p = img.get(Point2::new(img.size.x - x - 1, img.size.y - y - 1), 0);
            file.write_f32::<LittleEndian>(p.r.abs()).unwrap();
            file.write_f32::<LittleEndian>(p.g.abs()).unwrap();
            file.write_f32::<LittleEndian>(p.b.abs()).unwrap();
        }
    }
}

fn save_png(imgout_path_str: &str, img: &Bitmap) {
    // The image that we will render
    let mut image_ldr = DynamicImage::new_rgb8(img.size.x, img.size.y);
    for x in 0..img.size.x {
        for y in 0..img.size.y {
            let p = Point2::new(img.size.x - x - 1, y);
            image_ldr.put_pixel(x, y, img.get(p, 0).to_rgba())
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
) -> Bitmap {
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
fn integrate_image_plane<T: Integrator<Color>>(
    scene: &rustlight::scene::Scene,
    integrator: &T,
    nb_samples: usize,
) -> f32 {
    assert_ne!(nb_samples, 0);

    let mut sampler = ::rustlight::samplers::independent::IndependentSampler::default();
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

fn classical_mcmc_integration<T: Integrator<Color>>(
    scene: &rustlight::scene::Scene,
    nb_samples: usize,
    nb_threads: Option<usize>,
    large_prob: f32,
    int: T,
) -> Bitmap {
    ///////////// Prepare the pool for multiple thread
    let pool = match nb_threads {
        None => rayon::ThreadPoolBuilder::new(),
        Some(x) => rayon::ThreadPoolBuilder::new().num_threads(x),
    }.build()
        .unwrap();

    ///////////// Define the closure
    let sample = |s: &mut rustlight::samplers::mcmc::IndependentSamplerReplay| {
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
        samplers.push(rustlight::samplers::mcmc::IndependentSamplerReplay::default());
    }

    ///////////// Compute the rendering (with the number of samples)
    info!("Rendering...");
    let start = Instant::now();
    let progress_bar = Mutex::new(ProgressBar::new(samplers.len() as u64));
    let img = Mutex::new(Bitmap::new(1, Point2::new(0, 0), *scene.camera.size()));
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

            let mut my_img: Bitmap = Bitmap::new(1, Point2::new(0, 0), *scene.camera.size());
            (0..nb_samples_per_chains).into_iter().for_each(|_| {
                // Choose randomly between large and small perturbation
                s.large_step = s.rand() < large_prob;
                let mut proposed_state = sample(s);
                let accept_prob = (proposed_state.tf / current_state.tf).min(1.0);
                // Do waste reclycling
                current_state.weight += 1.0 - accept_prob;
                proposed_state.weight += accept_prob;
                if accept_prob > s.rand() {
                    my_img.accumulate(current_state.pix, current_state.color(), 0);
                    s.accept();
                    current_state = proposed_state;
                } else {
                    my_img.accumulate(proposed_state.pix, proposed_state.color(), 0);
                    s.reject();
                }
            });
            // Flush the last state
            my_img.accumulate(current_state.pix, current_state.color(), 0);

            my_img.scale(1.0 / (nb_samples_per_chains as f32));
            {
                img.lock().unwrap().accumulate_bitmap(&my_img);
                progress_bar.lock().unwrap().inc();
            }
        });
    });
    let mut img: Bitmap = img.into_inner().unwrap();
    let elapsed = start.elapsed();
    info!(
        "Elapsed: {} ms",
        (elapsed.as_secs() * 1_000) + (elapsed.subsec_nanos() / 1_000_000) as u64
    );

    // ==== Compute and scale to the normalization factor
    let img_avg = img.average_pixel(0);
    let img_avg_lum = (img_avg.r + img_avg.g + img_avg.b) / 3.0;
    img.scale(b / img_avg_lum);

    img
}

fn reconstruct(
    iterations: usize,
    img_size: Vector2<u32>,
    primal_image: &Bitmap,
    dx_image: &Bitmap,
    dy_image: &Bitmap,
    very_direct: &Bitmap,
) -> Bitmap {
    info!("Reconstruction...");
    let start = Instant::now();
    // Reconstruction (image-space covariate, uniform reconstruction)
    let mut current: Box<Bitmap> = Box::new(Bitmap::new(1, Point2::new(0, 0), img_size.clone()));
    let mut next: Box<Bitmap> = Box::new(Bitmap::new(1, Point2::new(0, 0), img_size.clone()));
    // 1) Init
    for y in 0..img_size.y {
        for x in 0..img_size.x {
            let pos = Point2::new(x, y);
            current.accumulate(pos, *primal_image.get(pos, 0), 0);
        }
    }
    for _iter in 0..iterations {
        // FIXME: Do it multi-threaded
        next.reset(); // Reset all to black
        for y in 0..img_size.y {
            for x in 0..img_size.x {
                let pos = Point2::new(x, y);
                let mut c = current.get(pos, 0).clone();
                let mut w = 1.0;
                if x > 0 {
                    let pos_off = Point2::new(x - 1, y);
                    c += current.get(pos_off, 0).clone() + dx_image.get(pos_off, 0).clone();
                    w += 1.0;
                }
                if x < img_size.x - 1 {
                    let pos_off = Point2::new(x + 1, y);
                    c += current.get(pos_off, 0).clone() - dx_image.get(pos, 0).clone();
                    w += 1.0;
                }
                if y > 0 {
                    let pos_off = Point2::new(x, y - 1);
                    c += current.get(pos_off, 0).clone() + dy_image.get(pos_off, 0).clone();
                    w += 1.0;
                }
                if y < img_size.y - 1 {
                    let pos_off = Point2::new(x, y + 1);
                    c += current.get(pos_off, 0).clone() - dy_image.get(pos, 0).clone();
                    w += 1.0;
                }
                c.scale(1.0 / w);
                next.accumulate(pos, c, 0);
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
    let mut image: Bitmap = Bitmap::new(1, Point2::new(0, 0), img_size.clone());
    for x in 0..img_size.x {
        for y in 0..img_size.y {
            let pos = Point2::new(x, y);
            let pix_value = next.get(pos, 0).clone() + very_direct.get(pos, 0).clone();
            image.accumulate(pos, pix_value, 0);
        }
    }
    image
}

/// This function decompose the multi channel bitmap to several bitmaps
/// These bitmap will be more easy to process
fn decompose_grad_color(img_grad: &Bitmap) -> (Bitmap, Bitmap, Bitmap, Bitmap) {
    let mut primal_image = Bitmap::new(1, Point2::new(0, 0), img_grad.size.clone());
    let mut dx_image = Bitmap::new(1, Point2::new(0, 0), img_grad.size.clone());
    let mut dy_image = Bitmap::new(1, Point2::new(0, 0), img_grad.size.clone());
    let mut very_direct = Bitmap::new(1, Point2::new(0, 0), img_grad.size.clone());

    for y in 0..img_grad.size.y {
        for x in 0..img_grad.size.x {
            let pos = Point2::new(x, y);
            primal_image.accumulate(pos, *img_grad.get(pos, ColorGradient::index("main")), 0);
            very_direct.accumulate(
                pos,
                *img_grad.get(pos, ColorGradient::index("very_direct")),
                0,
            );
            for (i, off) in GRADIENT_ORDER.iter().enumerate() {
                let pos_off: Point2<i32> = Point2::new(pos.x as i32 + off.x, pos.y as i32 + off.y);
                // FIXME: The primal image will be wrong
                // primal_image.accumulate_safe(pos_off, curr.radiances[i].clone());
                match GRADIENT_DIRECTION[i] {
                    GradientDirection::X(v) => match v {
                        1 => dx_image.accumulate(
                            pos,
                            *img_grad.get(pos, ColorGradient::index("dx_pos")),
                            0,
                        ),
                        -1 => dx_image.accumulate_safe(
                            pos_off,
                            (*img_grad.get(pos, ColorGradient::index("dx_neg"))) * -1.0,
                            0,
                        ),
                        _ => panic!("wrong displacement X"), // FIXME: Fix the enum
                    },
                    GradientDirection::Y(v) => match v {
                        1 => dy_image.accumulate(
                            pos,
                            *img_grad.get(pos, ColorGradient::index("dy_pos")),
                            0,
                        ),
                        -1 => dy_image.accumulate_safe(
                            pos_off,
                            (*img_grad.get(pos, ColorGradient::index("dy_neg"))) * -1.0,
                            0,
                        ),
                        _ => panic!("wrong displacement Y"),
                    },
                }
            }
        }
    }
    // Scale the throughtput image
    primal_image.scale(1.0 / 8.0); // TODO: Wrong at the corners, need to fix it

    (primal_image, dx_image, dy_image, very_direct)
}

fn gradient_domain_integration<T: Integrator<ColorGradient>>(
    scene: &rustlight::scene::Scene,
    nb_samples: usize,
    nb_threads: Option<usize>,
    int: T,
    iterations: usize,
) -> Bitmap {
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
    let (primal_image, dx_image, dy_image, very_direct) = decompose_grad_color(&img_grad);

    // Reconst
    reconstruct(
        iterations,
        img_grad.size,
        &primal_image,
        &dx_image,
        &dy_image,
        &very_direct,
    )
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
            SubCommand::with_name("gd-path")
                .about("gradient-domain path tracing")
                .arg(&max_arg)
                .arg(&min_arg)
                .arg(&iterations_arg),
        )
        .subcommand(
            SubCommand::with_name("gd-path-explicit")
                .about("gradient-domain path tracing with explicit path generation")
                .arg(&max_arg)
                .arg(&iterations_arg),
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
    let mut scene = rustlight::scene::Scene::new(&data, wk).expect("error when loading the scene");

    ///////////////// Tweak the image size
    {
        let image_scale = value_t_or_exit!(matches.value_of("image_scale"), f32);
        if image_scale != 1.0 {
            info!("Scale the image: {:?}", image_scale);
            assert!(image_scale != 0.0);
            scene.camera.scale_image(image_scale);
        }
    }

    ///////////////// Call the integrator for generating the bitmap
    let img = match matches.subcommand() {
        ("path-explicit", Some(m)) => {
            let max_depth = match_infinity(m.value_of("max").unwrap());

            classical_mc_integration(
                &scene,
                nb_samples,
                nb_threads,
                rustlight::integrators::path_explicit::IntegratorUniPath { max_depth },
            )
        }
        ("path", Some(m)) => {
            let max_depth = match_infinity(m.value_of("max").unwrap());
            let min_depth = match_infinity(m.value_of("min").unwrap());

            classical_mc_integration(
                &scene,
                nb_samples,
                nb_threads,
                rustlight::integrators::path::IntegratorPath {
                    max_depth,
                    min_depth,
                },
            )
        }
        ("pssmlt", Some(m)) => {
            let max_depth = match_infinity(m.value_of("max").unwrap());
            let min_depth = match_infinity(m.value_of("min").unwrap());
            let large_prob = value_t_or_exit!(m.value_of("large_prob"), f32);
            assert!(large_prob > 0.0 && large_prob <= 1.0);
            classical_mcmc_integration(
                &scene,
                nb_samples,
                nb_threads,
                large_prob,
                rustlight::integrators::path::IntegratorPath {
                    max_depth,
                    min_depth,
                },
            )
        }
        ("gd-path", Some(m)) => {
            let max_depth = match_infinity(m.value_of("max").unwrap());
            let min_depth = match_infinity(m.value_of("min").unwrap());
            let iterations = m.value_of("iterations").unwrap().parse::<usize>().unwrap();

            gradient_domain_integration(
                &scene,
                nb_samples,
                nb_threads,
                rustlight::integrators::path::IntegratorPath {
                    max_depth,
                    min_depth,
                },
                iterations,
            )
        }
        ("gd-path-explicit", Some(m)) => {
            let max_depth = match_infinity(m.value_of("max").unwrap());
            let iterations = m.value_of("iterations").unwrap().parse::<usize>().unwrap();

            gradient_domain_integration(
                &scene,
                nb_samples,
                nb_threads,
                rustlight::integrators::path_explicit::IntegratorUniPath { max_depth },
                iterations,
            )
        }
        ("ao", Some(m)) => {
            let dist = match_infinity(m.value_of("distance").unwrap());

            classical_mc_integration(
                &scene,
                nb_samples,
                nb_threads,
                rustlight::integrators::ao::IntegratorAO { max_distance: dist },
            )
        }
        ("direct", Some(m)) => classical_mc_integration(
            &scene,
            nb_samples,
            nb_threads,
            rustlight::integrators::direct::IntegratorDirect {
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
