use crate::emitter::*;
use crate::samplers::*;
use crate::scene::*;
use crate::structure::*;
use crate::tools::StepRangeInt;
use crate::Scale;

use cgmath::{Point2, Vector2};
use rayon;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use std;
use std::cmp;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

//////////////// Progress bar
/// PBR Wrapper for conditional compilation
#[cfg(feature = "progress-bar")]
struct ProgressBar {
    progress: pbr::ProgressBar<std::io::Stdout>
}
#[cfg(not(feature = "progress-bar"))]
struct ProgressBar {
}
impl ProgressBar {
    #[cfg(feature = "progress-bar")]
    fn new(n: u64) -> ProgressBar {
        ProgressBar {
            progress: pbr::ProgressBar::new(n)
        }
    }
    #[cfg(not(feature = "progress-bar"))]
    fn new(_: u64) -> ProgressBar {
        ProgressBar {}
    }

    #[cfg(feature = "progress-bar")]
    fn inc(&mut self) {
        self.progress.inc();
    }
    #[cfg(not(feature = "progress-bar"))]
    fn inc(&mut self) {}
}

//////////////// Helpers
/// Image block
/// for easy paralelisation over the thread
pub struct BufferCollection {
    pub pos: Point2<u32>,
    pub size: Vector2<u32>,
    pub values: HashMap<String, Bitmap>,
}

impl BufferCollection {
    /// Create a new Bitmap
    pub fn new(pos: Point2<u32>, size: Vector2<u32>, names: &[String]) -> BufferCollection {
        let mut bitmap = BufferCollection {
            pos,
            size,
            values: HashMap::new(),
        };

        for s in names {
            bitmap.register(s.clone());
        }
        bitmap
    }
    pub fn copy(
        pos: Point2<u32>,
        size: Vector2<u32>,
        other: &BufferCollection,
    ) -> BufferCollection {
        let mut bitmap = BufferCollection {
            pos,
            size,
            values: HashMap::new(),
        };
        for key in other.values.keys() {
            bitmap.register(key.clone());
        }
        bitmap
    }

    pub fn dump_all(&self, name: &str) {
        let output_ext = match std::path::Path::new(name).extension() {
            None => panic!("No file extension provided"),
            Some(x) => std::ffi::OsStr::to_str(x).expect("Issue to unpack the file"),
        };
        let mut trunc_name = name.to_string();
        trunc_name.truncate(name.len() - output_ext.len() - 1);
        for (key, value) in self.values.iter() {
            let new_name = format!("{}_{}.{}", trunc_name, key, output_ext);
            value.save(new_name.as_str());
        }
    }

    /// Register a name for a particular buffer
    pub fn register(&mut self, name: String) {
        self.values.insert(name, Bitmap::new(self.size));
    }

    pub fn register_mean_variance(
        &mut self,
        base_name: &str,
        o: &BufferCollection,
        buffers: &[String],
    ) {
        // Create buffers
        let mean_name = format!("{}_mean", base_name);
        let variance_name = format!("{}_variance", base_name);
        self.register(mean_name.clone());
        self.register(variance_name.clone());
        info!("average and variance: {:?}", buffers);
        for y in 0..o.size.y {
            for x in 0..o.size.x {
                // Compute mean
                let mut mean = Color::zero();
                for buffer in buffers {
                    mean += o.get(Point2::new(x, y), buffer);
                }
                mean.scale(1.0 / buffers.len() as f32);

                // Compute variance
                let mut variance = Color::zero();
                for buffer in buffers {
                    variance += (o.get(Point2::new(x, y), buffer) - mean).abs();
                }
                variance.scale(1.0 / buffers.len() as f32); // TODO: Check variance formula

                // Save the values
                self.accumulate(Point2::new(x, y), mean, &mean_name);
                self.accumulate(Point2::new(x, y), variance, &variance_name);
            }
        }
    }

    pub fn accumulate_bitmap_buffer(
        &mut self,
        o: &BufferCollection,
        name_org: &str,
        name_dest: &str,
    ) {
        let bitmap = self.values.get_mut(name_dest).unwrap();
        let other_bitmap = &o.values[name_org];
        bitmap.accumulate_bitmap(&other_bitmap, o.pos);
    }

    pub fn accumulate_bitmap(&mut self, o: &BufferCollection) {
        // This is special, it does not allowed to write twice the same pixels
        // This function is only when we
        for keys in o.values.keys() {
            self.accumulate_bitmap_buffer(o, keys, keys);
        }
    }

    pub fn accumulate(&mut self, p: Point2<u32>, f: Color, name: &str) {
        self.values.get_mut(name).unwrap().accumulate(p, f);
    }

    pub fn accumulate_safe(&mut self, p: Point2<i32>, f: Color, name: &str) {
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

    pub fn rename(&mut self, current_name: &str, new_name: &str) {
        let data = self.values.remove(current_name).unwrap();
        self.values.insert(new_name.to_string(), data);
    }

    pub fn get(&self, p: Point2<u32>, name: &str) -> Color {
        self.values[name].pixel(p)
    }

    pub fn reset(&mut self) {
        for val in self.values.values_mut() {
            val.clear();
        }
    }

    pub fn average_pixel(&self, name: &str) -> Color {
        self.values[name].average()
    }

    pub fn scale_buffer(&mut self, f: f32, name: &str) {
        self.values.get_mut(name).unwrap().scale(f);
    }

    pub fn save(&self, name: &str, filename: &str) {
        self.values[name].save(filename);
    }

    pub fn print_buffers_name(&self) {
        info!("buffers names: {:?}", self.values.keys());
    }
}

impl Scale<f32> for BufferCollection {
    fn scale(&mut self, f: f32) {
        assert!(f > 0.0);
        for val in self.values.values_mut() {
            val.scale(f);
        }
    }
}

/////////////// Integrators code
pub trait Integrator {
    fn compute(&mut self, _accel: &dyn Acceleration, scene: &Scene) -> BufferCollection {
        let buffernames = vec!["primal".to_string()];
        BufferCollection::new(Point2::new(0, 0), *scene.camera.size(), &buffernames)
    }
}
pub trait IntegratorGradient: Integrator {
    fn compute_gradients(&mut self, accel: &dyn Acceleration, scene: &Scene) -> BufferCollection;
    fn reconstruct(&self) -> &(dyn PoissonReconstruction + Sync);

    fn compute(&mut self, accel: &dyn Acceleration, scene: &Scene) -> BufferCollection {
        // Rendering the gradient informations
        info!("Gradient Rendering...");
        let start = Instant::now();
        let image = self.compute_gradients(accel, scene);
        let elapsed = start.elapsed();
        info!("Gradient Rendering Elapsed: {:?}", elapsed,);

        // Reconstruct the image
        info!("Reconstruction...");
        let start = Instant::now();
        let image = self.reconstruct().reconstruct(scene, &image);
        let elapsed = start.elapsed();
        info!("Reconstruction Elapsed: {:?}", elapsed,);

        image
    }
}
pub trait PoissonReconstruction {
    fn reconstruct(&self, scene: &Scene, est: &BufferCollection) -> BufferCollection;
    fn need_variance_estimates(&self) -> Option<usize>;
}
pub enum IntegratorType {
    Primal(Box<dyn Integrator>),
    Gradient(Box<dyn IntegratorGradient>),
}
impl IntegratorType {
    pub fn compute(&mut self, scene: &Scene) -> BufferCollection {
        info!("Build acceleration data structure...");

        // Naive Acceleration ...
        #[cfg(not(feature = "embree"))]
        let accel = NaiveAcceleration::new(scene);
        // or Embree ...
        // TODO: Need to found a work around due to the lifetime issue
        #[cfg(feature = "embree")]
        let embree_device = embree_rs::Device::new();
        #[cfg(feature = "embree")]
        let mut embree_scene = embree_rs::Scene::new(&embree_device);
        #[cfg(feature = "embree")]
        {
            for m in &scene.meshes {
                let mut tris = embree_rs::TriangleMesh::unanimated(
                    &embree_device,
                    m.indices.len(),
                    m.vertices.len(),
                );
                {
                    let mut verts = tris.vertex_buffer.map();
                    let mut tris = tris.index_buffer.map();
                    for i in 0..m.vertices.len() {
                        verts[i] = cgmath::Vector4::new(
                            m.vertices[i].x,
                            m.vertices[i].y,
                            m.vertices[i].z,
                            0.0,
                        );
                    }
    
                    for i in 0..m.indices.len() {
                        tris[i] = cgmath::Vector3::new(
                            m.indices[i].x as u32,
                            m.indices[i].y as u32,
                            m.indices[i].z as u32,
                        );
                    }
                }
                let mut tri_geom = embree_rs::Geometry::Triangle(tris);
                tri_geom.commit();
                embree_scene.attach_geometry(tri_geom);
            }
        }
        #[cfg(feature = "embree")]
        let accel = EmbreeAcceleration::new(scene, &embree_scene);

        info!("Run Integrator...");
        let start = Instant::now();

        let img = match self {
            IntegratorType::Primal(ref mut v) => v.compute(&accel, scene),
            IntegratorType::Gradient(ref mut v) => {
                IntegratorGradient::compute(v.as_mut(), &accel, scene)
            }
        };

        let elapsed = start.elapsed();
        info!("Elapsed Integrator: {} ms", elapsed.as_millis());

        img
    }
}

/////////////// Implementation gradients
pub trait IntegratorMC: Sync + Send {
    fn compute_pixel(
        &self,
        pix: (u32, u32),
        accel: &dyn Acceleration,
        scene: &Scene,
        sampler: &mut dyn Sampler,
        emitters: &EmitterSampler,
    ) -> Color;
}

pub fn generate_img_blocks(scene: &Scene, buffernames: &[String]) -> Vec<BufferCollection> {
    let mut image_blocks: Vec<BufferCollection> = Vec::new();
    for ix in StepRangeInt::new(0, scene.camera.size().x as usize, 16) {
        for iy in StepRangeInt::new(0, scene.camera.size().y as usize, 16) {
            let block = BufferCollection::new(
                Point2 {
                    x: ix as u32,
                    y: iy as u32,
                },
                Vector2 {
                    x: cmp::min(16, scene.camera.size().x - ix as u32),
                    y: cmp::min(16, scene.camera.size().y - iy as u32),
                },
                buffernames,
            );
            image_blocks.push(block);
        }
    }
    image_blocks
}

pub fn compute_mc<T: IntegratorMC + Integrator>(
    int: &T,
    accel: &dyn Acceleration,
    scene: &Scene,
) -> BufferCollection {
    // Here we can to the classical parallelisation
    assert_ne!(scene.nb_samples, 0);
    let buffernames = vec!["primal".to_string()];

    // Create rendering blocks
    let mut image_blocks = generate_img_blocks(scene, &buffernames);

    // Render the image blocks
    let progress_bar = Mutex::new(ProgressBar::new(image_blocks.len() as u64));
    let pool = generate_pool(scene);
    pool.install(|| {
        image_blocks.par_iter_mut().for_each(|im_block| {
            // image_blocks.iter_mut().for_each(|im_block| {
            let mut sampler = independent::IndependentSampler::default();
            let light_sampling = scene.emitters_sampler();
            for iy in 0..im_block.size.y {
                for ix in 0..im_block.size.x {
                    for _ in 0..scene.nb_samples {
                        let c = int.compute_pixel(
                            (ix + im_block.pos.x, iy + im_block.pos.y),
                            accel,
                            scene,
                            &mut sampler,
                            &light_sampling,
                        );
                        im_block.accumulate(Point2 { x: ix, y: iy }, c, &"primal".to_string());
                    }
                }
            }
            im_block.scale(1.0 / (scene.nb_samples as f32));

            {
                progress_bar.lock().unwrap().inc();
            }
        });
    });

    // Fill the image
    let mut image = BufferCollection::new(Point2::new(0, 0), *scene.camera.size(), &buffernames);
    for im_block in &image_blocks {
        image.accumulate_bitmap(im_block);
    }
    image
}

pub fn generate_pool(scene: &Scene) -> rayon::ThreadPool {
    match scene.nb_threads {
        None => rayon::ThreadPoolBuilder::new(),
        Some(x) => rayon::ThreadPoolBuilder::new().num_threads(x),
    }
    .build()
    .unwrap()
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
pub mod avg;
pub mod direct;
pub mod explicit;
pub mod gradient;
pub mod pssmlt;
