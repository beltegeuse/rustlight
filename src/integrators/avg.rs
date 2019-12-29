use crate::integrators::*;
use std;
use std::io::Write;

pub struct IntegratorAverage {
    pub time_out: Option<usize>, //< Time out in seconds
    pub integrator: IntegratorType,
}

impl Integrator for IntegratorAverage {
    fn compute(&mut self, accel: &dyn Acceleration, scene: &Scene) -> BufferCollection {
        // Get the output file type
        let output_ext = match std::path::Path::new(&scene.output_img_path).extension() {
            None => panic!("No file extension provided"),
            Some(x) => std::ffi::OsStr::to_str(x).expect("Issue to unpack the file"),
        };
        let mut base_output_img_path = scene.output_img_path.clone();
        base_output_img_path.truncate(scene.output_img_path.len() - output_ext.len() - 1);
        info!("Base output name: {:?}", base_output_img_path);

        // Open an CSV file for register the time
        let mut csv = std::fs::File::create(base_output_img_path.clone() + "_time.csv").unwrap();

        // Other values
        let mut bitmap: Option<BufferCollection> = None;
        let mut iteration = 1;
        let start = Instant::now();

        loop {
            let new_bitmap = match self.integrator {
                IntegratorType::Primal(ref mut v) => v.compute(accel, scene),
                IntegratorType::Gradient(ref mut v) => v.compute_gradients(accel, scene),
            };
            if iteration == 1 {
                bitmap = Some(new_bitmap);
            } else {
                bitmap.as_mut().unwrap().scale(iteration as f32);
                bitmap.as_mut().unwrap().accumulate_bitmap(&new_bitmap);
                bitmap.as_mut().unwrap().scale(1.0 / (iteration + 1) as f32);
            }

            // Save the bitmap for the current iteration
            let imgout_path_str = format!("{}_{}.{}", base_output_img_path, iteration, output_ext);
            match &self.integrator {
                IntegratorType::Primal(_) => bitmap
                    .as_ref()
                    .unwrap()
                    .save("primal", imgout_path_str.as_str()),
                IntegratorType::Gradient(ref v) => {
                    let start_recons = Instant::now();
                    let recons_img = v.reconstruct().reconstruct(scene, bitmap.as_ref().unwrap());
                    let elapsed_recons = start_recons.elapsed();
                    info!("Reconstruction time: {:?}", elapsed_recons);
                    // Save the bitmap for the current iteration
                    recons_img.save("primal", imgout_path_str.as_str());
                }
            };

            // Check the time elapsed when we started the rendering...
            let elapsed = start.elapsed();
            match self.time_out {
                None => info!("Total time (no timeout): {:?} secs", elapsed.as_secs()),
                Some(t) => info!("Total time: {:?} / {:?} secs", elapsed.as_secs(), t),
            }
            // Write the rendering time
            writeln!(csv, "{}.{},", elapsed.as_secs(), elapsed.subsec_millis()).unwrap();

            if self
                .time_out
                .map_or(false, |t| elapsed.as_secs() >= t as u64)
            {
                break;
            }
            // Update the number of iterations
            iteration += 1;
        }

        if let Some(bitmap) = bitmap {
            match &self.integrator {
                IntegratorType::Primal(_) => bitmap,
                IntegratorType::Gradient(v) => {
                    info!("Do the final reconstruction");
                    v.reconstruct().reconstruct(scene, &bitmap)
                }
            }
        } else {
            let buffernames = vec![String::from("primal")];
            BufferCollection::new(Point2::new(0, 0), *scene.camera.size(), &buffernames)
        }
    }
}
