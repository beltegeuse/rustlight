use cgmath::*;
use rand;
use std;

// my includes
use scene::*;
use math::*;
use structure::*;

pub trait Integrator {
    fn compute(&self, pix: (u32, u32), scene: &Scene) -> Color;
}

//////////// AO
pub struct IntergratorAO {
    pub max_distance : Option<f32>
}

impl Integrator for IntergratorAO {
    fn compute(&self, (ix,iy): (u32, u32), scene: &Scene) -> Color {
        let pix = (ix as f32 + rand::random::<f32>(), iy as f32 + rand::random::<f32>());
        let ray = scene.camera.generate(pix);

        // Do the intersection for the first path
        let intersection = scene.trace(&ray);
        if intersection.is_none() {
            return Color::zero();
        }
        let intersection = intersection.unwrap();

        // Compute an new direction
        // Note that we do not flip the normal automatically,
        // for the light definition (only one sided)
        let mut n_g = intersection.n_g;
        if n_g.dot(ray.d) > 0.0 {
            n_g = -n_g;
        }
        let frame = basis(n_g);
        let d = frame * cosine_sample_hemisphere(Point2::new(rand::random::<f32>(),
                                                             rand::random::<f32>()));

        // Check the new intersection distance
        let ray = Ray::new(intersection.p + d * 0.001, d);
        match scene.trace(&ray) {
            None => Color::one(1.0),
            Some(its) => {
                match self.max_distance {
                    None => Color::zero(),
                    Some(d) => if its.tfar > d { Color::one(1.0) } else { Color::zero() }
                }
            }
        }
    }
}

//////////// Direct
pub struct IntergratorDirect {
    pub nb_bsdf_samples : i32,
    pub nb_light_samples : i32
}

fn mis_weight(pdf_a: f32, pdf_b: f32) -> f32 {
    assert!(pdf_a != 0.0);
    assert!(pdf_a.is_finite());
    assert!(pdf_b.is_finite());
    pdf_a / (pdf_a + pdf_b)
}

impl Integrator for IntergratorDirect {
    fn compute(&self, (ix, iy): (u32, u32), scene: &Scene) -> Color {
        let pix = (ix as f32 + rand::random::<f32>(), iy as f32 + rand::random::<f32>());
        let ray = scene.camera.generate(pix);
        let mut l_i = Color::one(0.0);

        // Do the intersection for the first path
        let intersection = scene.trace(&ray);
        if intersection.is_none() { return l_i; }
        let intersection = intersection.unwrap();

        // Add the emission for the light intersection
        let init_mesh = &scene.meshes[intersection.geom_id as usize];
        l_i += &init_mesh.emission;

        // Get the good normal from the interaction
        let mut n_g = intersection.n_g.normalize();
        if n_g.dot(ray.d) > 0.0 { n_g = -n_g; }

        // Precompute for mis weights
        let weight_nb_bsdf = if self.nb_bsdf_samples == 0 { 0.0 } else { 1.0 / (self.nb_bsdf_samples as f32)};
        let weight_nb_light = if self.nb_light_samples == 0 { 0.0 } else { 1.0 / (self.nb_light_samples as f32)};

        /////////////////////////////////
        // Light sampling
        /////////////////////////////////
        // Explict connect to the light source
        for _ in 0..self.nb_light_samples {
            let light_record = scene.sample_light(&intersection.p,
                                                  rand::random::<f32>(),
                                                  (rand::random::<f32>(), rand::random::<f32>()));
            if light_record.is_valid() && scene.visible(&intersection.p, &light_record.p) {
                // Compute the contribution of direct lighting
                let bsdf_val: Color = init_mesh.bsdf.clone() * (n_g.dot(light_record.d).max(0.0) / std::f32::consts::PI);
                let pdf_bsdf = n_g.dot(light_record.d).max(0.0) / std::f32::consts::PI;

                // Compute MIS weights
                let weight_light = mis_weight(light_record.pdf * weight_nb_light,
                                              pdf_bsdf * weight_nb_bsdf);
                l_i += weight_light
                    * bsdf_val
                    * weight_nb_light
                    * light_record.weight;
            }
        }

        /////////////////////////////////
        // BSDF sampling
        /////////////////////////////////
        // Compute an new direction (diffuse)
        let frame = basis(n_g);
        for _ in 0..self.nb_bsdf_samples {
            let d_local = cosine_sample_hemisphere(Point2::new(
                rand::random::<f32>(),
                rand::random::<f32>()));
            let d = frame * d_local;

            // Generate the new ray and do the intersection
            let ray = Ray::new(intersection.p.clone(), d);
            let intersection = scene.trace(&ray);
            if intersection.is_none() {
                continue;
            }
            let intersection = intersection.unwrap();
            let intersected_mesh = &scene.meshes[intersection.geom_id as usize];

            // Check that we have intersected a light or not
            if intersected_mesh.is_light() {
                let cos_light = intersection.n_g.dot(ray.d).max(0.0);
                if cos_light == 0.0 {
                    continue;
                }

                // Compute MIS weights
                // FIXME: the pdf for selecting the light
                let geom_light = cos_light / (intersection.tfar * intersection.tfar);
                let pdf_light_sa = intersected_mesh.pdf() * geom_light;
                let pdf_bsdf_sa = n_g.dot(d).max(0.0) / std::f32::consts::PI;
                let weight_bsdf = mis_weight(pdf_bsdf_sa * weight_nb_bsdf,
                                             pdf_light_sa * weight_nb_light);

                l_i += weight_bsdf * init_mesh.bsdf.clone() * (&intersected_mesh.emission) * weight_nb_light;
            }
        }

        l_i
    }
}

//////////// Path tracing
pub struct IntergratorPath {
    pub max_depth : i32
}

impl Integrator for IntergratorPath {
    fn compute(&self, (ix, iy): (u32, u32), scene: &Scene) -> Color {
        // Generate the first ray
        let pix = (ix as f32 + rand::random::<f32>(), iy as f32 + rand::random::<f32>());
        let mut ray = scene.camera.generate(pix);
        let mut l_i = Color::one(0.0);
        let mut throughput = Color::one(1.0);

        // Check if we have a intersection with the primary ray
        let intersection_opt = scene.trace(&ray);
        if intersection_opt.is_none() {
            return l_i;
        }
        let mut intersection = intersection_opt.unwrap();

        let mut depth = 1;
        while depth < self.max_depth {
            // Add the emission for the light intersection
            let hit_mesh = &scene.meshes[intersection.geom_id as usize];
            if depth == 1 {
                l_i += &hit_mesh.emission;
            }

            // Get the good normal from the interaction
            let mut n_g = intersection.n_g.normalize();
            if n_g.dot(ray.d) > 0.0 {
                n_g = -n_g;
            }

            /////////////////////////////////
            // Light sampling
            /////////////////////////////////
            // Explict connect to the light source
            let light_record = scene.sample_light(&intersection.p,
                                                  rand::random::<f32>(),
                                                  (rand::random::<f32>(), rand::random::<f32>()));
            if light_record.is_valid() && scene.visible(&intersection.p, &light_record.p) {
                // Compute the contribution of direct lighting
                let bsdf_val: Color = hit_mesh.bsdf.clone() * (n_g.dot(light_record.d).max(0.0) / std::f32::consts::PI);
                let pdf_bsdf = n_g.dot(light_record.d).max(0.0) / std::f32::consts::PI;

                // Compute MIS weights
                let weight_light = mis_weight(light_record.pdf,
                                              pdf_bsdf);
                l_i += weight_light
                    * throughput.clone()
                    * bsdf_val
                    * light_record.weight;
            }

            /////////////////////////////////
            // BSDF sampling
            /////////////////////////////////
            // Compute an new direction (diffuse)
            let frame = basis(n_g);
            let d_local = cosine_sample_hemisphere(Point2::new(rand::random::<f32>(),
                                                               rand::random::<f32>()));
            let d = frame * d_local;

            // Update the throughput
            throughput *= &hit_mesh.bsdf;

            // Generate the new ray and do the intersection
            ray = Ray::new(intersection.p, d);
            let intersection_opt = scene.trace(&ray);
            if intersection_opt.is_none() {
                return l_i;
            }
            intersection = intersection_opt.unwrap();
            let next_mesh = &scene.meshes[intersection.geom_id as usize];

            // Check that we have intersected a light or not
            let cos_light = intersection.n_g.dot(ray.d).max(0.0);
            let cos_surf = n_g.dot(d).max(0.0);
            if next_mesh.is_light() && cos_light != 0.0 && cos_surf != 0.0 {
                // Compute MIS weights
                // FIXME: the pdf for selecting the light
                let geom_light =  cos_light / (intersection.tfar * intersection.tfar);
                let pdf_light_sa = next_mesh.pdf() * geom_light;
                let pdf_bsdf_sa = cos_surf / std::f32::consts::PI;

                let weight_bsdf = mis_weight(pdf_bsdf_sa, pdf_light_sa);

                l_i +=  (throughput.clone()) * (&next_mesh.emission) * weight_bsdf;
            }

            // Increase the depth of the current path
            depth += 1;
        }

        l_i
    }
}