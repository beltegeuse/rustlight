use integrators::*;
use std;
use tools;

pub struct IntegratorAverage {
    pub time_out: Option<usize>, //< Time out in seconds
    pub output_csv: bool,
    pub integrator: Box<Integrator>,
}

impl Integrator for IntegratorAverage {
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
            let new_bitmap = self.integrator.compute(scene);
            if iteration == 1 {
                bitmap = Some(new_bitmap);
            } else {
                bitmap.as_mut().unwrap().scale(iteration as f32);
                bitmap.as_mut().unwrap().accumulate_bitmap(&new_bitmap);
                bitmap.as_mut().unwrap().scale(1.0 / (iteration + 1) as f32);
            }

            // Save the bitmap for the current iteration
            let imgout_path_str = format!("{}_{}.{}", base_output_img_path, iteration, output_ext);
            tools::save(
                imgout_path_str.as_str(),
                bitmap.as_ref().unwrap(),
                "primal".to_string(),
            );

            // Check the time elapsed when we started the rendering...
            let elapsed = start.elapsed();
            match self.time_out {
                None => info!("Total time (no timeout): {:?} secs", elapsed.as_secs()),
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
            let buffernames = vec![String::from("primal")];
            bitmap = Some(Bitmap::new(
                Point2::new(0, 0),
                *scene.camera.size(),
                &buffernames,
            ));
        }
        return bitmap.unwrap();
    }
}
