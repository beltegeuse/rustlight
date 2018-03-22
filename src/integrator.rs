use cgmath::*;
use std::ops::AddAssign;
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
                    let light_pdf = scene.direct_pdf(LightSamplingPDF::new(scene,
                                                                           &ray, &intersection));

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
    pub max_depth : Option<u32>,
    pub min_depth : Option<u32>,
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
        while self.max_depth.map_or(true, |max| depth < max) {
            // Check if we go the right orientation
            if intersection.n_g.dot(ray.d) > 0.0 {
                return l_i;
            }

            // Add the emission for the light intersection
            let hit_mesh = &scene.meshes[intersection.geom_id as usize];
            if self.min_depth.map_or(true, |min| depth >= min) && depth == 1 {
                l_i += &hit_mesh.emission; // TODO: Add throughput
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
                if self.min_depth.map_or(true, |min| depth >= min) {
                    l_i += weight_light
                        * throughput
                        * hit_mesh.bsdf.eval(&d_in_local, &d_out_local)
                        * light_record.weight;
                }
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
                let light_pdf = scene.direct_pdf(LightSamplingPDF::new(scene,
                                                                       &ray, &intersection));

                let weight_bsdf = mis_weight(sampled_bsdf.pdf, light_pdf);
                if self.min_depth.map_or(true, |min| depth >= min) {
                    l_i += throughput * (&next_mesh.emission) * weight_bsdf;
                }
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

#[derive(Clone, Debug, Copy)]
pub struct ColorGradient {
    pub very_direct: Color,
    pub main: Color,
    pub radiances: [Color; 4],
    pub gradients: [Color; 4],
}
pub enum GradientDirection {
    X(i32),
    Y(i32)
}
pub static GRADIENT_ORDER: [Point2<i32>; 4] = [Point2 {x:0, y:1 },
    Point2{x: 0, y: -1},
    Point2{x:1, y: 0},
    Point2{x: -1, y:0}];
pub static GRADIENT_DIRECTION: [GradientDirection; 4] = [
    GradientDirection::Y(1), GradientDirection::Y(-1),
    GradientDirection::X(1), GradientDirection::X(-1)
];

impl Default for ColorGradient {
    fn default() -> Self {
        ColorGradient {
            very_direct: Color::zero(),
            main: Color::zero(),
            radiances: [Color::zero(); 4],
            gradients: [Color::zero(); 4],
        }
    }
}
impl AddAssign<ColorGradient> for ColorGradient {
    fn add_assign(&mut self, other: ColorGradient) {
        self.very_direct += other.very_direct;
        self.main += other.main;
        for i in 0..self.gradients.len() {
            self.radiances[i] += other.radiances[i];
            self.gradients[i] += other.gradients[i];
        }
    }
}
impl Scale<f32> for ColorGradient {
    fn scale(&mut self, v: f32) {
        self.very_direct.scale(v);
        self.main.scale(v);
        for i in 0..self.gradients.len() {
            self.radiances[i].scale(v);
            self.gradients[i].scale(v);
        }
    }
}
struct RayStateData {
    pub pdf: f32,
    pub ray: Ray,
    pub its: Intersection,
    pub throughput: Color,
}
enum RayState {
    NotConnected(RayStateData),
    RecentlyConnected(RayStateData),
    Connected(RayStateData), // FIXME: Do we need to store all the data?
    Dead,
}

impl RayState {
    pub fn check_normal(self) -> RayState {
        // FIXME: Change how this works .... to avoid duplicated code
        match self {
            RayState::NotConnected(e) =>
                if e.its.n_g.dot(e.ray.d) > 0.0 {
                    RayState::Dead
                } else {
                    RayState::NotConnected(e)
                },
            RayState::RecentlyConnected(e) =>
                if e.its.n_g.dot(e.ray.d) > 0.0 {
                    RayState::Dead
                } else {
                    RayState::RecentlyConnected(e)
                },
            RayState::Connected(e) => {
                // FIXME: Maybe not a good idea...
                // FIXME: As the shift path may be not used anymore
                assert!(e.its.n_g.dot(e.ray.d) <= 0.0);
                RayState::Connected(e)
            }
            RayState::Dead => RayState::Dead,
        }
    }

    pub fn apply_russian_roulette(&mut self, rr_prob: f32) {
        match self {
            &mut RayState::Dead => {},
            &mut RayState::NotConnected(ref mut e) |
            &mut RayState::Connected(ref mut e) |
            &mut RayState::RecentlyConnected(ref mut e) => {
                e.throughput /= rr_prob;
            }
        }
    }

    pub fn new((x, y): (f32, f32), off: &Point2<i32>, scene: &Scene) -> RayState {
        let pix = (x + off.x as f32, y + off.y as f32);
        if  pix.0 < 0.0 || pix.0 > (scene.camera.size().x as f32) ||
            pix.1 < 0.0 || pix.1 > (scene.camera.size().y as f32) {
            return RayState::Dead;
        }

        let ray = scene.camera.generate(pix);
        let its = match scene.trace(&ray) {
            Some(x) => x,
            None => return RayState::Dead,
        };

        RayState::NotConnected(RayStateData {
            pdf: 1.0,
            ray,
            its,
            throughput: Color::one(),
        })
    }
}

impl Integrator<ColorGradient> for IntegratorPath {
    fn compute(&self, (ix, iy): (u32, u32), scene: &Scene, sampler: &mut Sampler) -> ColorGradient {
        let mut l_i = ColorGradient::default();
        let pix =  (ix as f32 + sampler.next(), iy as f32 + sampler.next());
        let mut main = match RayState::new(pix, &Point2::new(0,0), scene) {
            RayState::NotConnected(x) => x,
            _ => return l_i,
        };
        let mut offsets: Vec<RayState> = {
            GRADIENT_ORDER.iter().map(|e| RayState::new(pix, e, &scene)).collect()
        };

        const MIS_POWER: i32 = 1;

        // For now, just replay the random numbers
        let mut depth: u32 = 1;
        while self.max_depth.is_none() || (depth < self.max_depth.unwrap()) {
            // Check if we go the right orientation
            // -- main path
            if main.its.n_g.dot(main.ray.d) > 0.0 {
                return l_i;
            }
            offsets = offsets.into_iter().map(|e| e.check_normal()).collect();

            // Add the emission for the light intersection
            let main_hit_mesh = &scene.meshes[main.its.geom_id as usize];
            if self.min_depth.map_or(true, |min| depth >= min) && depth == 1 {
                l_i.very_direct += &main_hit_mesh.emission; // TODO: Add throughput
            }


            // Construct local frame
            let main_frame = basis(main.its.n_g);
            let main_d_in_local = main_frame.to_local(-main.ray.d);

            /////////////////////////////////
            // Light sampling
            /////////////////////////////////
            // Explict connect to the light source
            {
                // FIXME: Zero contributon handling: BSDF and others
                let (r_sel_rand, r_rand, uv_rand) = (sampler.next(),
                                                     sampler.next(),
                                                     sampler.next2d());
                let main_light_record = scene.sample_light(&main.its.p,
                                                      r_sel_rand, r_rand, uv_rand);
                let main_light_visible = scene.visible(&main.its.p, &main_light_record.p);
                let main_emitter_rad = if main_light_visible { main_light_record.weight } else { Color::zero() };
                let main_d_out_local = main_frame.to_local(main_light_record.d);
                // Evaluate BSDF values
                let main_bsdf_value = main_hit_mesh.bsdf.eval(&main_d_in_local, &main_d_out_local); // f(...) * cos(...)
                let main_bsdf_pdf = if main_light_visible { main_hit_mesh.bsdf.pdf(&main_d_in_local, &main_d_out_local) } else { 0.0 };
                // Cache PDF / throughput values
                let main_weight_num = (main.pdf * main_light_record.pdf).powi(MIS_POWER);
                let main_weight_dem = (main.pdf.powi(MIS_POWER)) *
                    (main_light_record.pdf.powi(MIS_POWER) + main_bsdf_pdf.powi(MIS_POWER));
                let main_contrib = main.throughput * main_bsdf_value * main_emitter_rad;
                // Cache geometric informations
                let main_geom_dsquared = (main.its.p - main_light_record.p).magnitude2();
                let main_geom_cos_light = main_light_record.n.dot(main_light_record.d);
                if main_light_record.pdf != 0.0 {
                    // FIXME: Double check this condition. Normally, it will be fine
                    // FIXME: But pdf = 0 for the main path does not necessary imply
                    // FIXME: 0 probability for the shift path, no?
                    for (i,offset) in offsets.iter().enumerate() {
                        let (shift_weight_dem, shift_contrib) = match offset {
                            &RayState::Dead => { ( main_weight_num / (0.0001 + main_weight_dem), Color::zero()) },
                            &RayState::Connected(ref s) => {
                                // FIXME: See if we can simplify the structure, as we need to know:
                                //  - throughput
                                //  - pdf
                                // only
                                let shift_weight_dem = s.pdf.powi(MIS_POWER) *
                                    (main_light_record.pdf.powi(MIS_POWER) + main_bsdf_pdf.powi(MIS_POWER));
                                let shift_contrib = s.throughput * main_bsdf_value * main_emitter_rad;
                                (shift_weight_dem, shift_contrib)
                            },
                            &RayState::RecentlyConnected(ref s) => {
                                // Need to re-evaluate the BSDF as the incomming direction is different
                                // FIXME: We only need to know:
                                //  - throughput
                                //  - pdf
                                //  - incomming direction (in world space)
                                let shift_d_in_global = (s.its.p - main.its.p).normalize();
                                let shift_d_in_local = main_frame.to_local(shift_d_in_global);
                                // BSDF
                                let shift_bsdf_pdf = if main_light_visible && shift_d_in_local.z > 0.0 {
                                    main_hit_mesh.bsdf.pdf(&shift_d_in_local, &main_d_out_local)
                                } else { 0.0 };
                                let shift_bsdf_value = if shift_d_in_local.z > 0.0 { main_hit_mesh.bsdf.eval(&shift_d_in_local, &main_d_out_local) } else { Color::zero() };
                                // Compute and return
                                let shift_weight_dem = (s.pdf).powi(MIS_POWER) *
                                    (main_light_record.pdf.powi(MIS_POWER) + shift_bsdf_pdf.powi(MIS_POWER));
                                let shift_contrib = s.throughput * shift_bsdf_value * main_emitter_rad;
                                (shift_weight_dem, shift_contrib)
                            }
                            &RayState::NotConnected(ref s) => {
                                // Get intersection informations
                                let shift_hit_mesh = &scene.meshes[s.its.geom_id as usize];
                                let shift_frame = basis(s.its.n_g);
                                let shift_d_in_local = shift_frame.to_local(-s.ray.d);
                                // FIXME: We need to check the light source type in order to continue or not
                                // FIXME: the ray tracing...

                                // Sample the light from the point
                                let shift_light_record = scene.sample_light(&s.its.p,
                                                                            r_sel_rand,
                                                                            r_rand,
                                                                            uv_rand);
                                let shift_light_visible = scene.visible(&s.its.p,
                                                                        &shift_light_record.p);
                                let shift_emitter_rad = if shift_light_visible { shift_light_record.weight * (shift_light_record.pdf / main_light_record.pdf) } else { Color::zero() };
                                let shift_d_out_local = shift_frame.to_local(shift_light_record.d);
                                // BSDF
                                let shift_bsdf_value = shift_hit_mesh.bsdf.eval(&shift_d_in_local, &shift_d_out_local);
                                let shift_bsdf_pdf = if shift_light_visible { shift_hit_mesh.bsdf.pdf(&shift_d_in_local, &shift_d_out_local) } else { 0.0 };
                                // Compute Jacobian: Here the ratio of geometry terms
                                let jacobian = ( shift_light_record.n.dot(shift_light_record.d) * main_geom_dsquared).abs()
                                    / (0.0001 + (main_geom_cos_light * (s.its.p - shift_light_record.p).magnitude2()).abs()); // FIXME: Inf jacobian?
                                // Bake the final results
                                let shift_weight_dem = (jacobian * s.pdf).powi(MIS_POWER) *
                                    (shift_light_record.pdf.powi(MIS_POWER) + shift_bsdf_pdf.powi(MIS_POWER));
                                let shift_contrib = jacobian * s.throughput * shift_bsdf_value * shift_emitter_rad;
                                (shift_weight_dem, shift_contrib)
                            },
                        };

                        if self.min_depth.map_or(true, |min| depth >= min) {
                            //let weight = if shift_weight_dem == 0.0 { 1.0 } else { 0.5 };
                            let weight = main_weight_num / (main_weight_dem + shift_weight_dem);
                            assert!(weight.is_finite());
                            l_i.main += main_contrib * weight;
                            l_i.radiances[i] += shift_contrib * weight;
                            l_i.gradients[i] += (shift_contrib - main_contrib) * weight;
                        }
                    }
                }
            }

            /////////////////////////////////
            // BSDF sampling
            /////////////////////////////////
            // Compute an new direction (diffuse)
            let main_sampled_bsdf = match main_hit_mesh.bsdf.sample(&main_d_in_local, sampler.next2d()) {
                Some(x) => x,
                None => return l_i,
            };

            // Generate the new ray and do the intersection
            let main_d_out_global = main_frame.to_world(main_sampled_bsdf.d);
            main.ray = Ray::new(main.its.p, main_d_out_global);
            main.its = match scene.trace(&main.ray) {
                Some(x) => x,
                None => return l_i,
            };
            let main_next_mesh = &scene.meshes[main.its.geom_id as usize];

            // Check that we have intersected a light or not
            let (main_light_pdf, main_emitter_rad) = {
                if main_next_mesh.is_light() && main.its.n_g.dot(-main.ray.d).max(0.0) != 0.0 {
                    let light_pdf = scene.direct_pdf(LightSamplingPDF::new(scene,
                                                                           &main.ray, &main.its));
                    (light_pdf, main_next_mesh.emission.clone())
                } else {
                    (0.0, Color::zero())
                }
            };

            // Update the main path
            let main_pdf_pred = main.pdf; // FIXME: need to compute the MIS weight after this point
            main.throughput *= &(main_sampled_bsdf.weight);
            main.pdf *= main_sampled_bsdf.pdf;
            if main.pdf == 0.0 || main.throughput.is_zero() {
                return l_i;
            }

            let main_weight_num = (main_pdf_pred * main_sampled_bsdf.pdf).powi(MIS_POWER);
            let main_weight_dem = (main_pdf_pred.powi(MIS_POWER)) *
                ((main_sampled_bsdf.pdf.powi(MIS_POWER)) + main_light_pdf.powi(MIS_POWER));
            let main_contrib = main.throughput * main_emitter_rad;

            offsets = offsets.into_iter()
                .enumerate().map(|(i,offset)| {
                let (shift_weight_dem, shift_contrib, new_state) = match offset {
                    RayState::Dead => ( 0.0, Color::zero(), RayState::Dead),
                    RayState::Connected(mut s) => {
                        let shift_pdf_pred = s.pdf;
                        // Update the shifted path
                        s.throughput *= &(main_sampled_bsdf.weight);
                        s.pdf *= main_sampled_bsdf.pdf;
                        // Compute the return values
                        let shift_weight_dem = shift_pdf_pred.powi(MIS_POWER) *
                            (main_sampled_bsdf.pdf.powi(MIS_POWER) + main_light_pdf.powi(MIS_POWER));
                        let shift_contrib = s.throughput * main_emitter_rad;
                        ( shift_weight_dem, shift_contrib, RayState::Connected(s))
                    },
                    RayState::RecentlyConnected(mut s) => {
                        let shift_d_in_global = (s.its.p - main.its.p).normalize();
                        let shift_d_in_local = main_frame.to_local(shift_d_in_global);
                        let main_d_out_local = main_frame.to_local(main.ray.d);
                        // BSDF
                        let shift_bsdf_pdf = if shift_d_in_local.z > 0.0 { main_hit_mesh.bsdf.pdf(&shift_d_in_local, &main_d_out_local) } else { 0.0 };
                        let shift_bsdf_value = if shift_d_in_local.z > 0.0 { main_hit_mesh.bsdf.eval(&shift_d_in_local, &main_d_out_local) } else { Color::zero() };
                        // Update main path
                        let shift_pdf_pred = s.pdf;
                        s.throughput *= &(shift_bsdf_value / main_sampled_bsdf.pdf);
                        s.pdf *= shift_bsdf_pdf;
                        // Compute and return
                        let shift_weight_dem = shift_pdf_pred.powi(MIS_POWER) *
                            (shift_bsdf_pdf.powi(MIS_POWER) + main_light_pdf.powi(MIS_POWER));
                        let shift_contrib = s.throughput * main_emitter_rad;
                        ( shift_weight_dem, shift_contrib, RayState::Connected(s))
                    }
                    RayState::NotConnected(mut s) => {
                        // FIXME: Always do a reconnection here
                        // FIXME: Implement half-vector copy
                        if !scene.visible(&s.its.p, &main.its.p) {
                            ( 0.0, Color::zero(), RayState::Dead ) // FIXME: Found a way to do it in an elegant way
                        } else {
                            // The current mesh that we do the intersection
                            let shift_hit_mesh = &scene.meshes[s.its.geom_id as usize];
                            // Compute the ratio of geometry factors
                            let shift_d_out_global = (main.its.p - s.its.p).normalize();
                            let shift_frame = basis(s.its.n_g);
                            let shift_d_out_local = shift_frame.to_local(shift_d_out_global);
                            let shift_d_in_local = shift_frame.to_local(-s.ray.d);
                            // FIXME: Inf jacobian?
                            let jacobian = ( main.its.n_g.dot(-shift_d_out_global) * main.its.t.powi(2)).abs()
                                / (0.0001 + (main.its.n_g.dot(-main.ray.d) * (s.its.p - main.its.p).magnitude2()).abs());
                            // BSDF
                            let shift_bsdf_value = shift_hit_mesh.bsdf.eval(&shift_d_in_local, &shift_d_out_local);
                            let shift_bsdf_pdf= shift_hit_mesh.bsdf.pdf(&shift_d_in_local, &shift_d_out_local);
                            // FIXME: Dead path? if pdf == 0.0, maybe add inside a check
                            // Update shift path
                            let shift_pdf_pred = s.pdf;
                            s.throughput *= &((shift_bsdf_value * jacobian) / main_sampled_bsdf.pdf);
                            s.pdf *= shift_bsdf_pdf * jacobian;

                            // Two case:
                            // - the main are on a emitter, need to do MIS
                            // - the main are not on a emitter, just do a reconnection
                            let (shift_emitter_rad, shift_emitter_pdf) = if main_light_pdf == 0.0 {
                                // The base path did not hit a light source
                                // FIXME: Do not use the trick of 0 PDF
                                (Color::zero(), 0.0)
                            } else {
                                let shift_emitter_pdf = scene.direct_pdf(LightSamplingPDF {
                                    mesh: main_next_mesh,
                                    o: s.its.p,
                                    p: main.its.p,
                                    n: main.its.n_g,
                                    dir: shift_d_out_global,
                                });
                                // FIXME: We return without the cos as the light
                                // FIXME: does not change, does it true for non uniform light?
                                (main_emitter_rad.clone(), shift_emitter_pdf)
                            };

                            // Return the shift path updated + MIS weights
                            let shift_weight_dem = shift_pdf_pred.powi(MIS_POWER)
                                * (shift_bsdf_pdf.powi(MIS_POWER) + shift_emitter_pdf.powi(MIS_POWER) );
                            let shift_contrib = s.throughput * shift_emitter_rad;
                            ( shift_weight_dem, shift_contrib, RayState::RecentlyConnected(s))
                        }
                    },
                };
                // Update the contributions
                if self.min_depth.map_or(true, |min| depth >= min) {
                    //let weight = if shift_weight_dem == 0.0 { 1.0 } else { 0.5 };
                    let weight = main_weight_num / (main_weight_dem + shift_weight_dem);
                    assert!(weight.is_finite());
                    l_i.main += main_contrib * weight;
                    l_i.radiances[i] += shift_contrib * weight;
                    l_i.gradients[i] += (shift_contrib - main_contrib) * weight;
                }
                // Return the new state
                new_state
            }).collect();

            // Russian roulette
            let rr_pdf = main.throughput.channel_max().min(0.95);
            if rr_pdf < sampler.next() {
                break;
            }
            main.throughput /= rr_pdf;
            offsets.iter_mut().for_each(|o| o.apply_russian_roulette(rr_pdf));

            // Increase the depth of the current path
            depth += 1;
        }

        l_i
    }
}



