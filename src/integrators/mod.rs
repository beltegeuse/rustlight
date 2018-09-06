use samplers::*;
use scene::*;

pub trait Integrator: Sync + Send {
    fn compute<S: Sampler>(&self, scene: &Scene, sampler: &mut S) -> Bitmap;
    fn nb_samples(&self) -> usize;
}
pub trait IntegratorMC: Integrator {
    fn compute_pixel<S: Sampler>(&self, pix: (u32, u32), scene: &Scene, sampler: &mut S) -> Color;
}
impl Integrator for IntegratorMC {
    fn compute(&self, scene: &Scene, sampler: &mut S) -> Bitmap {
        // Here we can to the classical parallelisation
        assert_ne!(self.nb_samples(), 0);

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
        let mut image = Bitmap::new(Point2::new(0, 0), *scene.camera.size());
        for im_block in &image_blocks {
            image.accumulate_bitmap(im_block);
        }
        image
    }
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
