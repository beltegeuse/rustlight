use integrators::gradient::*;
use std;
use tools;

pub struct IntegratorGradientAverage {
    pub time_out: Option<usize>, //< Time out in seconds
    pub output_csv: bool,
    pub integrator: Box<IntegratorGradient>,
}

impl Integrator for IntegratorGradientAverage {
    fn compute(&mut self, scene: &Scene) -> Bitmap {
        // Get the output file type
        let output_ext = match std::path::Path::new(&scene.output_img_path).extension() {
            None => panic!("No file extension provided"),
            Some(x) => std::ffi::OsStr::to_str(x).expect("Issue to unpack the file"),
        };
        let mut base_output_img_path = scene.output_img_path.clone();
        base_output_img_path.truncate(scene.output_img_path.len() - output_ext.len() - 1);
        info!("Base output name: {:?}", base_output_img_path);

        // Other values
        let mut bitmap: Option<Bitmap> = None;
        let mut iteration = 1;
        let start = Instant::now();

        loop {
            let new_bitmap = self.integrator.compute_gradients(scene);
            if iteration == 1 {
                bitmap = Some(new_bitmap);
            } else {
                bitmap.as_mut().unwrap().scale(iteration as f32);
                bitmap.as_mut().unwrap().accumulate_bitmap(&new_bitmap);
                bitmap.as_mut().unwrap().scale(1.0 / (iteration + 1) as f32);
            }

            // Do the reconstruction
            let recons_img = self
                .integrator
                .reconstruct()
                .reconstruct(scene, bitmap.as_ref().unwrap());

            // Save the bitmap for the current iteration
            let imgout_path_str =
                base_output_img_path.clone() + "_" + &iteration.to_string() + "." + output_ext;
            tools::save(imgout_path_str.as_str(), &recons_img, "primal");

            // Check the time elapsed when we started the rendering...
            let elapsed = start.elapsed();
            match self.time_out {
                None => info!("Total time (no timeout): {:?}", elapsed),
                Some(t) => info!("Total time: {:?} / {:?} secs", elapsed.as_secs(), t),
            }
            if self
                .time_out
                .map_or(false, |t| elapsed.as_secs() >= t as u64)
            {
                break;
            }
            // Update the number of iterations
            iteration += 1;
        }

        if bitmap.is_none() {
            let buffernames = vec!["primal"];
            bitmap = Some(Bitmap::new(
                Point2::new(0, 0),
                *scene.camera.size(),
                &buffernames,
            ));
        }
        return bitmap.unwrap();
    }
}
