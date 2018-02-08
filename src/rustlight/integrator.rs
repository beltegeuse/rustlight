use cgmath::*;
use rustlight::scene::*;
use rustlight::math::*;
use rustlight::structure::*;
use embree; // FIXME: Remove this
use rand;

/////////////////////////
// Functions
pub fn compute_ao((ix,iy): (u32,u32),scene: &Scene) -> Option<Color> {
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

    let d_local = cosine_sample_hemisphere(Point2::new(rand::random::<f32>(),
                                                       rand::random::<f32>()));
    let d = frame * d_local;

    let o = intersection.p + d * 0.001;
    let mut embree_ray_new = embree::rtcore::Ray::new(&o,
                                                      &d);
    scene.embree.occluded(&mut embree_ray_new);
    if embree_ray_new.hit() {
        None
    } else {
        Some(Color::one(1.0))
    }
}

pub fn compute_direct((ix,iy): (u32,u32),scene: &Scene) -> Option<Color> {
    let pix = (ix as f32 + rand::random::<f32>(), iy as f32 + rand::random::<f32>());
    let ray = scene.camera.generate(pix);

    // Do the intersection for the first path
    let intersection = scene.trace(&ray);
    if intersection.is_none() {
        return None;
    }
    let intersection = intersection.unwrap();
    let init_mesh = &scene.meshes[intersection.geom_id as usize];

    // Compute an new direction (diffuse)
    let mut n_g_normalized = intersection.n_g.normalize();
    if n_g_normalized.dot(ray.d) > 0.0 {
        n_g_normalized = -n_g_normalized;
    }
    let frame = basis(n_g_normalized);
    let d_local = cosine_sample_hemisphere(Point2::new(rand::random::<f32>(),
                                                       rand::random::<f32>()));
    let d = frame * d_local;

    let o = intersection.p + d * 0.001;
    let ray = Ray::new(o, d);

    // Do the other intersection
    let intersection = scene.trace(&ray);
    if intersection.is_none() {
        return None;
    }
    let intersection = intersection.unwrap();
    let intersected_mesh = &scene.meshes[intersection.geom_id as usize];

    if !intersected_mesh.is_light() {
        return None;
    }
    if intersection.n_g.dot(ray.d) <= 0.0 {
        return None;
    }

    Some(init_mesh.bsdf.clone() * (&intersected_mesh.emission))
}