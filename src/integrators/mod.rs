use cgmath::{Point2, Vector2};
use pbr::ProgressBar;
use rayon;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use samplers::*;
use scene::*;
use std::cmp;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;
use structure::Color;
use tools::StepRangeInt;
use Scale;

//////////////// Helpers
/// Image block
/// for easy paralelisation over the thread
pub struct Bitmap {
    pub pos: Point2<u32>,
    pub size: Vector2<u32>,
    pub values: HashMap<String, Vec<Color>>,
}
unsafe impl Send for Bitmap {}

impl Bitmap {
    /// Create a new Bitmap
    pub fn new(pos: Point2<u32>, size: Vector2<u32>, names: &Vec<String>) -> Bitmap {
        let mut bitmap = Bitmap {
            pos,
            size,
            values: HashMap::new(),
        };

        for s in names {
            bitmap.register(&s);
        }
        bitmap
    }
    pub fn copy(pos: Point2<u32>, size: Vector2<u32>, other: &Bitmap) -> Bitmap {
        let mut bitmap = Bitmap {
            pos,
            size,
            values: HashMap::new(),
        };
        for key in other.values.keys() {
            bitmap.register(key);
        }
        bitmap
    }

    /// Register a name for a particular buffer
    pub fn register(&mut self, name: &String) {
        self.values.insert(
            name.to_string(),
            vec![Color::default(); (self.size.x * self.size.y) as usize],
        );
    }

    pub fn accumulate_bitmap(&mut self, o: &Bitmap) {
        // This is special, it does not allowed to write twice the same pixels
        // This function is only when we

        for keys in o.values.keys() {
            let mut pixels = self.values.get_mut(keys).unwrap();
            let other_pixels = &o.values[keys];
            for y in 0..o.size.y {
                for x in 0..o.size.x {
                    let p = Point2::new(o.pos.x + x, o.pos.y + y);
                    let index = (p.y * self.size.y + p.x) as usize;
                    let index_other = (y * o.size.y + x) as usize;
                    pixels[index] += other_pixels[index_other];
                }
            }
        }
    }

    pub fn accumulate(&mut self, p: Point2<u32>, f: Color, name: &String) {
        assert!(p.x < self.size.x);
        assert!(p.y < self.size.y);
        let index = (p.y * self.size.y + p.x) as usize;
        self.values.get_mut(name).unwrap()[index] += f;
    }

    pub fn accumulate_safe(&mut self, p: Point2<i32>, f: Color, name: &String) {
        if p.x >= 0 && p.y >= 0 && p.x < (self.size.x as i32) && p.y < (self.size.y as i32) {
            self.accumulate(
                Point2 {
                    x: p.x as u32,
                    y: p.y as u32,
                },
                f,
                name,
            );
        }
    }

    pub fn get(&self, p: Point2<u32>, name: &String) -> &Color {
        assert!(p.x < self.size.x);
        assert!(p.y < self.size.y);
        &self.values[name][(p.y * self.size.y + p.x) as usize]
    }

    pub fn reset(&mut self) {
        for (_, val) in self.values.iter_mut() {
            val.iter_mut().for_each(|x| *x = Color::default());
        }
    }

    pub fn average_pixel(&self, name: &String) -> Color {
        let mut s = Color::default();
        self.values[name].iter().for_each(|x| s += x.clone());
        s.scale(1.0 / self.values[name].len() as f32);
        s
    }
}

impl Scale<f32> for Bitmap {
    fn scale(&mut self, f: f32) {
        assert!(f > 0.0);
        for (_, val) in self.values.iter_mut() {
            val.iter_mut().for_each(|v| v.scale(f));
        }
    }
}

/////////////// Integrators code
pub trait Integrator: Sync + Send {
    fn compute(&self, scene: &Scene) -> Bitmap {
        let buffernames = vec!["primal".to_string()];
        Bitmap::new(Point2::new(0, 0), *scene.camera.size(), &buffernames)
    }
    fn preprocess(&mut self, scene: &Scene) {
    }
}
pub trait IntegratorMC: Sync + Send {
    fn compute_pixel(&self, pix: (u32, u32), scene: &Scene, sampler: &mut Sampler) -> Color;
}

pub fn compute_mc<T: IntegratorMC + Integrator>(int: &T, scene: &Scene) -> Bitmap {
    // Here we can to the classical parallelisation
    assert_ne!(scene.nb_samples(), 0);
    let buffernames = vec!["primal".to_string()];

    // Create rendering blocks
    let mut image_blocks: Vec<Bitmap> = Vec::new();
    for ix in StepRangeInt::new(0, scene.camera.size().x as usize, 16) {
        for iy in StepRangeInt::new(0, scene.camera.size().y as usize, 16) {
            let mut block = Bitmap::new(
                Point2 {
                    x: ix as u32,
                    y: iy as u32,
                },
                Vector2 {
                    x: cmp::min(16, scene.camera.size().x - ix as u32),
                    y: cmp::min(16, scene.camera.size().y - iy as u32),
                },
                &buffernames,
            );
            image_blocks.push(block);
        }
    }

    // Render the image blocks
    let progress_bar = Mutex::new(ProgressBar::new(image_blocks.len() as u64));
    image_blocks.par_iter_mut().for_each(|im_block| {
        let mut sampler = independent::IndependentSampler::default();
        for ix in 0..im_block.size.x {
            for iy in 0..im_block.size.y {
                for _ in 0..scene.nb_samples() {
                    let c = int.compute_pixel(
                        (ix + im_block.pos.x, iy + im_block.pos.y),
                        scene,
                        &mut sampler,
                    );
                    im_block.accumulate(Point2 { x: ix, y: iy }, c, &"primal".to_string());
                }
            }
        }
        im_block.scale(1.0 / (scene.nb_samples() as f32));

        {
            progress_bar.lock().unwrap().inc();
        }
    });

    // Fill the image
    let mut image = Bitmap::new(Point2::new(0, 0), *scene.camera.size(), &buffernames);
    for im_block in &image_blocks {
        image.accumulate_bitmap(im_block);
    }
    image
}

pub fn run_integrator<T: Integrator + Send + Sync>(
    scene: &Scene,
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
    let img = pool.install(|| int.compute(scene));
    let elapsed = start.elapsed();
    info!(
        "Elapsed: {} ms",
        (elapsed.as_secs() * 1_000) + (elapsed.subsec_nanos() / 1_000_000) as u64
    );

    return img;
}

/// Power heuristic for path tracing or direct lighting
pub fn mis_weight(pdf_a: f32, pdf_b: f32) -> f32 {
    if pdf_a == 0.0 {
        warn!("MIS weight requested for 0.0 pdf");
        return 0.0;
    }
    assert!(pdf_a.is_finite());
    assert!(pdf_b.is_finite());
    let w = pdf_a.powi(2) / (pdf_a.powi(2) + pdf_b.powi(2));
    if w.is_finite() {
        w
    } else {
        warn!("Not finite MIS weight for: {} and {}", pdf_a, pdf_b);
        0.0
    }
}

pub mod ao;
pub mod direct;
pub mod explicit;
pub mod path;
pub mod prelude;
pub mod pssmlt;
