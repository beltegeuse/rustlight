use cgmath::*;
use rand;
use std;

// my includes
use scene::*;
use math::*;
use structure::*;

/////////////////////////
// Functions
pub fn compute_ao((ix, iy): (u32, u32), scene: &Scene) -> Option<Color> {
    let pix = (ix as f32 + rand::random::<f32>(), iy as f32 + rand::random::<f32>());
    let ray = scene.camera.generate(pix);

    // Do the intersection for the first path
    let intersection = scene.trace(&ray);
    if intersection.is_none() {
        return None;
    }
    let intersection = intersection.unwrap();

    // Compute an new direction
    let mut n_g = intersection.n_g;
    if n_g.dot(ray.d) > 0.0 {
        n_g = -n_g;
    }
    let frame = basis(n_g);

    let d = frame * cosine_sample_hemisphere(Point2::new(rand::random::<f32>(),
                                                         rand::random::<f32>()));

    let ray = Ray::new(intersection.p + d * 0.001, d);
    if scene.hit(&ray) {
        None
    } else {
        Some(Color::one(1.0))
    }
}

fn mis_weight(pdf_a: f32, pdf_b: f32) -> f32 {
    pdf_a / (pdf_a + pdf_b)
}

pub fn compute_direct((ix, iy): (u32, u32), scene: &Scene) -> Color {
    let pix = (ix as f32 + rand::random::<f32>(), iy as f32 + rand::random::<f32>());
    let ray = scene.camera.generate(pix);
    let mut l_i = Color::one(0.0);

    // Do the intersection for the first path
    let intersection = scene.trace(&ray);
    if intersection.is_none() {
        return l_i;
    }
    let intersection = intersection.unwrap();

    // Add the emission for the light intersection
    let init_mesh = &scene.meshes[intersection.geom_id as usize];
    l_i += &init_mesh.emission;

    // Get the good normal from the interaction
    let mut n_g = intersection.n_g.normalize();
    if n_g.dot(ray.d) > 0.0 {
        n_g = -n_g;
    }

    /////////////////////////////////
    // Light sampling
    /////////////////////////////////
    // Explict connect to the light source
    //TODO: Select the light source
    let emitter_id = scene.emitters[0];
    let emitter = &scene.meshes[emitter_id];
    let (p_light, n_light, pdf_light) = emitter.sample(rand::random::<f32>(), (rand::random::<f32>(), rand::random::<f32>()));

    if scene.visible(&intersection.p, &p_light) {
        let mut d: Vector3<f32> = p_light - intersection.p;
        let dist = d.magnitude();
        d /= dist;

        // Compute the contribution of direct lighting
        let bsdf_val: Color = init_mesh.bsdf.clone() * (n_g.dot(d).max(0.0) / std::f32::consts::PI);
        let geom_light = (n_light.dot(-d).max(0.0) / (dist * dist));

        // Compute MIS weights
        let pdf_light_sa = pdf_light.clone() * geom_light;
        let pdf_bsdf_sa = n_g.dot(d).max(0.0) / std::f32::consts::PI;
        let weight_light = mis_weight(pdf_light_sa, pdf_bsdf_sa);

        // FIXME: use lightPDF
        l_i += weight_light
            * bsdf_val
            * (&emitter.emission)
            * (1.0 / pdf_light)
            * geom_light;
    }

    /////////////////////////////////
    // BSDF sampling
    /////////////////////////////////
    // Compute an new direction (diffuse)
    let frame = basis(n_g);
    let d_local = cosine_sample_hemisphere(Point2::new(rand::random::<f32>(),
                                                       rand::random::<f32>()));
    let d = frame * d_local;

    // Generate the new ray and do the intersection
    let o = intersection.p + d * 0.001;
    let ray = Ray::new(o, d);
    let intersection = scene.trace(&ray);
    if intersection.is_none() {
        return l_i;
    }
    let intersection = intersection.unwrap();
    let intersected_mesh = &scene.meshes[intersection.geom_id as usize];

    // Check that we have intersected a light or not
    if !intersected_mesh.is_light() {
        return l_i;
    }
    let cos_light = intersection.n_g.dot(ray.d).max(0.0);
    if cos_light == 0.0 {
        return l_i;
    }

    // Compute MIS weights
    // FIXME: the pdf for selecting the light
    let geom_light =  cos_light / (intersection.tfar * intersection.tfar);
    let pdf_light_sa = intersected_mesh.pdf() * geom_light;
    let pdf_bsdf_sa = n_g.dot(d).max(0.0) / std::f32::consts::PI;

    let weight_bsdf = mis_weight(pdf_bsdf_sa, pdf_light_sa);

    l_i += weight_bsdf * init_mesh.bsdf.clone() * (&intersected_mesh.emission);
    l_i
}