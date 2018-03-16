use cgmath::*;

// my includes
use scene::*;
use math::*;
use structure::*;
use sampler::*;
use embree_rs::ray::Intersection;

pub trait Integrator<T> {
    fn compute(&self, pix: (u32, u32), scene: &Scene, sampler: &mut Sampler) -> T;
}

//////////// AO
pub struct IntergratorAO {
    pub max_distance : Option<f32>
}

impl Integrator<Color> for IntergratorAO {
    fn compute(&self, (ix,iy): (u32, u32), scene: &Scene, sampler: &mut Sampler) -> Color {
        let pix = (ix as f32 + sampler.next(), iy as f32 + sampler.next());
        let ray = scene.camera.generate(pix);

        // Do the intersection for the first path
        let intersection = match scene.trace(&ray)  {
            Some(its) => its,
            None => return Color::zero(),
        };

        // Compute an new direction
        // Note that we do not flip the normal automatically,
        // for the light definition (only one sided)
        if intersection.n_g.dot(ray.d) > 0.0 {
            return Color::zero();
        }

        let frame = basis(intersection.n_g);
        let d_local = cosine_sample_hemisphere(sampler.next2d());
        let d_world = frame.to_world(d_local);

        // Check the new intersection distance
        let ray = Ray::new(intersection.p, d_world);
        match scene.trace(&ray) {
            None => Color::one(),
            Some(its) => {
                match self.max_distance {
                    None => Color::zero(),
                    Some(d) => if its.t > d { Color::one() } else { Color::zero() }
                }
            }
        }
    }
}

//////////// Direct
pub struct IntergratorDirect {
    pub nb_bsdf_samples : u32,
    pub nb_light_samples : u32
}

fn mis_weight(pdf_a: f32, pdf_b: f32) -> f32 {
    assert!(pdf_a != 0.0);
    assert!(pdf_a.is_finite());
    assert!(pdf_b.is_finite());
    pdf_a / (pdf_a + pdf_b)
}

impl Integrator<Color> for IntergratorDirect {
    fn compute(&self, (ix, iy): (u32, u32), scene: &Scene, sampler: &mut Sampler) -> Color {
        let pix = (ix as f32 + sampler.next(), iy as f32 + sampler.next());
        let ray = scene.camera.generate(pix);
        let mut l_i = Color::zero();

        // Do the intersection for the first path
        let intersection = match scene.trace(&ray)  {
            Some(its) => its,
            None => return l_i,
        };

        // Check if we go the right orientation
        if intersection.n_g.dot(ray.d) > 0.0 {
            return l_i;
        }

        // Project incoming direction in local space
        let frame = basis(intersection.n_g);
        let d_in_local = frame.to_local(-ray.d);

        // Add the emission for the light intersection
        let init_mesh = &scene.meshes[intersection.geom_id as usize];
        l_i += &init_mesh.emission;

        // Precompute for mis weights
        let weight_nb_bsdf = if self.nb_bsdf_samples == 0 { 0.0 } else { 1.0 / (self.nb_bsdf_samples as f32)};
        let weight_nb_light = if self.nb_light_samples == 0 { 0.0 } else { 1.0 / (self.nb_light_samples as f32)};

        /////////////////////////////////
        // Light sampling
        /////////////////////////////////
        // Explict connect to the light source
        for _ in 0..self.nb_light_samples {
            let light_record = scene.sample_light(&intersection.p,
                                                  sampler.next(),
                                                  sampler.next(),
                                                  sampler.next2d());
            let d_out_local = frame.to_local(light_record.d);
            if light_record.is_valid() && scene.visible(&intersection.p, &light_record.p) && d_out_local.z > 0.0 {
                // Compute the contribution of direct lighting
                let pdf_bsdf = init_mesh.bsdf.pdf(&d_in_local, &d_out_local);

                // Compute MIS weights
                let weight_light = mis_weight(light_record.pdf * weight_nb_light,
                                              pdf_bsdf * weight_nb_bsdf);
                l_i += weight_light
                    * init_mesh.bsdf.eval(&d_in_local, &d_out_local)
                    * weight_nb_light
                    * light_record.weight;
            }
        }

        /////////////////////////////////
        // BSDF sampling
        /////////////////////////////////
        // Compute an new direction (diffuse)
        for _ in 0..self.nb_bsdf_samples {
            let d_in_local = frame.to_local(-ray.d);
            if let Some(sampled_bsdf) = init_mesh.bsdf.sample(&d_in_local, sampler.next2d()) {
                // Generate the new ray and do the intersection
                let d_out_world = frame.to_world(sampled_bsdf.d);
                let ray = Ray::new(intersection.p, d_out_world);
                let intersection = match scene.trace(&ray) {
                    Some(x) => x,
                    None => continue,
                };
                let intersected_mesh = &scene.meshes[intersection.geom_id as usize];

                // Check that we have intersected a light or not
                if intersected_mesh.is_light() && intersection.n_g.dot(-ray.d) > 0.0 {
                    // FIXME: Found an elegant way to retreive incomming Le
                    let light_pdf = scene.direct_pdf(&ray, &intersection);

                    // Compute MIS weights
                    let weight_bsdf = mis_weight(sampled_bsdf.pdf * weight_nb_bsdf,
                                                 light_pdf * weight_nb_light);

                    l_i += weight_bsdf * sampled_bsdf.weight * (&intersected_mesh.emission) * weight_nb_bsdf;
                }
            }
        }

        l_i
    }
}

////////////// Path tracing
pub struct IntegratorPath {
    pub max_depth : Option<u32>
}
impl Integrator<Color> for IntegratorPath {
    fn compute(&self, (ix, iy): (u32, u32), scene: &Scene, sampler: &mut Sampler) -> Color {
        // Generate the first ray
        let pix = (ix as f32 + sampler.next(), iy as f32 + sampler.next());
        let mut ray = scene.camera.generate(pix);
        let mut l_i = Color::zero();
        let mut throughput = Color::one();

        // Check if we have a intersection with the primary ray
        let mut intersection = match scene.trace(&ray) {
            Some(x) => x,
            None => return l_i,
        };

        let mut depth: u32 = 1;
        while self.max_depth.is_none() || (depth < self.max_depth.unwrap()) {
            // Check if we go the right orientation
            if intersection.n_g.dot(ray.d) > 0.0 {
                return l_i;
            }

            // Add the emission for the light intersection
            let hit_mesh = &scene.meshes[intersection.geom_id as usize];
            if depth == 1 {
                l_i += &hit_mesh.emission;
            }

            // Construct local frame
            let frame = basis(intersection.n_g);
            let d_in_local = frame.to_local(-ray.d);

            /////////////////////////////////
            // Light sampling
            /////////////////////////////////
            // Explict connect to the light source
            let light_record = scene.sample_light(&intersection.p,
                                                  sampler.next(),
                                                  sampler.next(),
                                                  sampler.next2d());
            let d_out_local = frame.to_local(light_record.d);
            if light_record.is_valid() && scene.visible(&intersection.p, &light_record.p) && d_out_local.z > 0.0 {
                // Compute the contribution of direct lighting
                let pdf_bsdf = hit_mesh.bsdf.pdf(&d_in_local, &d_out_local);

                // Compute MIS weights
                let weight_light = mis_weight(light_record.pdf, pdf_bsdf);
                l_i += weight_light
                    * throughput.clone()
                    * hit_mesh.bsdf.eval(&d_in_local, &d_out_local)
                    * light_record.weight;
            }

            /////////////////////////////////
            // BSDF sampling
            /////////////////////////////////
            // Compute an new direction (diffuse)
            let sampled_bsdf = match hit_mesh.bsdf.sample(&d_in_local, sampler.next2d()) {
                Some(x) => x,
                None => return l_i,
            };

            // Update the throughput
            throughput *= &sampled_bsdf.weight;

            // Generate the new ray and do the intersection
            let d_out_global = frame.to_world(sampled_bsdf.d);
            ray = Ray::new(intersection.p, d_out_global);
            intersection = match scene.trace(&ray) {
                Some(x) => x,
                None => return l_i,
            };
            let next_mesh = &scene.meshes[intersection.geom_id as usize];

            // Check that we have intersected a light or not
            let cos_light = intersection.n_g.dot(-ray.d).max(0.0); // FIXME
            if next_mesh.is_light() && cos_light != 0.0 {
                let light_pdf = scene.direct_pdf(&ray, &intersection);

                let weight_bsdf = mis_weight(sampled_bsdf.pdf, light_pdf);
                l_i +=  (throughput.clone()) * (&next_mesh.emission) * weight_bsdf;
            }

            // Russian roulette
            let rr_pdf = throughput.channel_max().min(0.95);
            if rr_pdf < sampler.next() {
                break;
            }
            throughput /= rr_pdf;
            // Increase the depth of the current path
            depth += 1;
        }

        l_i
    }
}

pub struct IntegratorGradientPath {
    pub max_depth: Option<u32>,
}
pub struct ColorGradient {
    pub main: Color,
    pub gradients: [Color; 4],
}
impl Default for ColorGradient {
    fn default() -> Self {
        ColorGradient {
            main: Color::zero(),
            gradients: [Color::zero(); 4],
        }
    }
}
struct RayState {
    pub pdf: f32,
    pub ray: Ray,
    pub its: Intersection,
    pub throughput: Color,
}
impl RayState {
    pub fn new((x, y): (f32, f32), (xoff, yoff): (i32, i32), scene: &Scene) -> Option<RayState> {
        let pix = (x + xoff as f32, y + yoff as f32);
        if  pix.0 < 0.0 || pix.0 > (scene.camera.size().x as f32) ||
            pix.1 < 0.0 || pix.1 > (scene.camera.size().y as f32) {
            return None;
        }

        let ray = scene.camera.generate(pix);
        let its = match scene.trace(&ray) {
            Some(x) => x,
            None => return None,
        };

        Some(RayState {
            pdf: 1.0, // FIXME: need somehow use cos^3
            ray,
            its,
            throughput: Color::one(),
        })
    }
}

impl Integrator<ColorGradient> for IntegratorPath {
    fn compute(&self, (ix, iy): (u32, u32), scene: &Scene, sampler: &mut Sampler) -> ColorGradient {
        let l_i = ColorGradient::default();
        let pix =  (ix as f32 + sampler.next(), iy as f32 + sampler.next());
        let main = match RayState::new(pix, (0,0), &scene) {
            None => return l_i,
            Some(x) => x,
        };
        let offset: Vec<Option<RayState>> = {
            let indices = [(0,1), (0, -1), (1, 0), (-1, 0)];
            indices.iter().map(|e| RayState::new(pix, *e, &scene)).collect()
        };

        // For now, just replay the random numbers
        

        return l_i;

    }
}



