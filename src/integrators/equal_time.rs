use crate::integrators::*;
use std;

pub struct IntegratorEqualTime {
    pub target_time_ms: u128,
    pub integrator: IntegratorType,
}

impl Integrator for IntegratorEqualTime {
    fn compute(
        &mut self,
        sampler: &mut dyn Sampler,
        accel: &dyn Acceleration,
        scene: &Scene,
    ) -> BufferCollection {
        // Other values
        let mut bitmap: Option<BufferCollection> = None;
        let mut iteration = 1;
        let mut time_rendering = std::time::Duration::new(0, 0);

        loop {
            let start = Instant::now();
            let new_bitmap = match self.integrator {
                IntegratorType::Primal(ref mut v) => v.compute(sampler, accel, scene),
                IntegratorType::Gradient(ref mut v) => v.compute_gradients(sampler, accel, scene),
            };
            if iteration == 1 {
                bitmap = Some(new_bitmap);
            } else {
                let averaging = match &self.integrator {
                    IntegratorType::Primal(v) => v.averaging(),
                    IntegratorType::Gradient(v) => v.averaging(),
                };
                if averaging {
                    bitmap.as_mut().unwrap().accumulate_bitmap(&new_bitmap);
                } else {
                    bitmap = Some(new_bitmap);
                }
            }
            time_rendering += start.elapsed();
            if time_rendering.as_millis() >= self.target_time_ms {
                break;
            }

            // Update the number of iterations
            iteration += 1;
        }

        bitmap.as_mut().unwrap().scale(1.0 / (iteration) as f32);
        info!("Number iter: {}", iteration);
        info!("Number spp: {}", iteration * scene.nb_samples);

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
