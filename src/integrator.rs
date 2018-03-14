use cgmath::*;

// my includes
use scene::*;
use math::*;
use structure::*;
use sampler::*;

pub trait Integrator {
    fn compute(&self, pix: (u32, u32), scene: &Scene, sampler: &mut Sampler) -> Color;
}

//////////// AO
pub struct IntergratorAO {
    pub max_distance : Option<f32>
}

impl Integrator for IntergratorAO {
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

impl Integrator for IntergratorDirect {
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
            let (d_out_local, bsdf_pdf, bsdf_value) = init_mesh.bsdf.sample(&d_in_local, sampler.next2d());

            // Generate the new ray and do the intersection
            let d_out_world = frame.to_world(d_out_local);
            let ray = Ray::new(intersection.p, d_out_world);
            let intersection = match scene.trace(&ray) {
                Some(x) => x,
                None => continue,
            };
            let intersected_mesh = &scene.meshes[intersection.geom_id as usize];

            // Check that we have intersected a light or not
            if intersected_mesh.is_light() && intersection.n_g.dot(-ray.d) > 0.0 {
                // FIXME: Found an elegant way to retreive incomming Le
                let light_pdf = scene.direct_pdf(&ray,&intersection);

                // Compute MIS weights
                let weight_bsdf = mis_weight(bsdf_pdf * weight_nb_bsdf,
                                                 light_pdf * weight_nb_light);

                l_i += weight_bsdf * bsdf_value * (&intersected_mesh.emission) * weight_nb_bsdf;
            }
        }

        l_i
    }
}

////////////// Path tracing
pub struct IntergratorPath {
    pub max_depth : i32
}

impl Integrator for IntergratorPath {
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

        let mut depth = 1;
        while depth < self.max_depth {
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
            let (d_out_local, bsdf_pdf, bsdf_value) = hit_mesh.bsdf.sample(&d_in_local, sampler.next2d());
            if bsdf_value.is_zero() {
                return l_i;
            }

            // Update the throughput
            throughput *= &bsdf_value;

            // Generate the new ray and do the intersection
            let d_out_global = frame.to_world(d_out_local);
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

                let weight_bsdf = mis_weight(bsdf_pdf, light_pdf);
                l_i +=  (throughput.clone()) * (&next_mesh.emission) * weight_bsdf;
            }

            // Increase the depth of the current path
            depth += 1;
        }

        l_i
    }
}