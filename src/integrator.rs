use cgmath::*;
use rand;

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
    let mut n_g_normalized = intersection.n_g.normalize();
    if n_g_normalized.dot(ray.d) > 0.0 {
        n_g_normalized = -n_g_normalized;
    }
    let frame = basis(n_g_normalized);

    let d = frame * cosine_sample_hemisphere(Point2::new(rand::random::<f32>(),
                                                         rand::random::<f32>()));

    let ray = Ray::new(intersection.p + d * 0.001, d);
    if scene.hit(&ray) {
        None
    } else {
        Some(Color::one(1.0))
    }
}

pub fn compute_direct((ix, iy): (u32, u32), scene: &Scene) -> Option<Color> {
    let pix = (ix as f32 + rand::random::<f32>(), iy as f32 + rand::random::<f32>());
    let ray = scene.camera.generate(pix);

    // Do the intersection for the first path
    let intersection = scene.trace(&ray);
    if intersection.is_none() {
        return None;
    }
    let intersection = intersection.unwrap();
    let init_mesh = &scene.meshes[intersection.geom_id as usize];

    // Explict connect to the light source
    //TODO: Select the light source
    let emitter_id = scene.emitters[0];
    let emitter = &scene.meshes[emitter_id];
    let (p_light, pdf_light) = emitter.sample(rand::random::<f32>(), (rand::random::<f32>(), rand::random::<f32>()));

    if !scene.visible(&intersection.p, &p_light) {
        return None;
    }

    // Compute the contribution
    let mut d: Vector3<f32> = p_light - intersection.p;
    let dist = d.magnitude();
    d /= dist;

    let mut n_g_normalized = intersection.n_g.normalize();
    if n_g_normalized.dot(ray.d) > 0.0 {
        n_g_normalized = -n_g_normalized;
    }

    let bsdf_val: Color = init_mesh.bsdf.clone() * n_g_normalized.dot(d).max(0.0);

    // FIXME: missing normal for the emitter
    // FIXME: use lightPDF
    Some(bsdf_val * (&emitter.emission) * (1.0 / (dist * dist)))


//    // Compute an new direction (diffuse)
//    let mut n_g_normalized = intersection.n_g.normalize();
//    if n_g_normalized.dot(ray.d) > 0.0 {
//        n_g_normalized = -n_g_normalized;
//    }
//    let frame = basis(n_g_normalized);
//    let d_local = cosine_sample_hemisphere(Point2::new(rand::random::<f32>(),
//                                                       rand::random::<f32>()));
//    let d = frame * d_local;
//
//    let o = intersection.p + d * 0.001;
//    let ray = Ray::new(o, d);
//
//    // Do the other intersection
//    let intersection = scene.trace(&ray);
//    if intersection.is_none() {
//        return None;
//    }
//    let intersection = intersection.unwrap();
//    let intersected_mesh = &scene.meshes[intersection.geom_id as usize];
//
//    if !intersected_mesh.is_light() {
//        return None;
//    }
//    if intersection.n_g.dot(ray.d) <= 0.0 {
//        return None;
//    }
//
//    Some(init_mesh.bsdf.clone() * (&intersected_mesh.emission))
}