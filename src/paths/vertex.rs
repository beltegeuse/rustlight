use bsdfs::*;
use cgmath::*;
use samplers::*;
use scene::*;
use structure::*;

#[derive(Clone)]
pub struct Edge {
    pub dist: Option<f32>,
    pub d: Vector3<f32>,
}

#[derive(Clone)]
pub struct SensorVertex {
    pub uv: Point2<f32>,
    pub pos: Point3<f32>,
    // FIXME: Add as Option
    pub pdf: f32, // FIXME: Add as Option
}

#[derive(Clone)]
pub struct SurfaceVertex<'a> {
    pub its: Intersection<'a>,
    pub throughput: Color,
    pub sampled_bsdf: Option<SampledDirection>,
    pub rr_weight: f32,
}

#[derive(Clone)]
pub enum Vertex<'a> {
    Sensor(SensorVertex),
    Surface(SurfaceVertex<'a>),
}

impl<'a> Vertex<'a> {
    pub fn new_sensor_vertex(uv: Point2<f32>, pos: Point3<f32>) -> Vertex<'a> {
        Vertex::Sensor(SensorVertex { uv, pos, pdf: 1.0 })
    }

    pub fn generate_next<S: Sampler>(
        &mut self,
        scene: &'a Scene,
        sampler: Option<&mut S>,
    ) -> (Option<Edge>, Option<Vertex<'a>>) {
        match *self {
            Vertex::Sensor(ref mut v) => {
                let ray = scene.camera.generate(v.uv);
                let its = match scene.trace(&ray) {
                    Some(its) => its,
                    None => {
                        return (
                            Some(Edge {
                                dist: None,
                                d: ray.d,
                            }),
                            None,
                        )
                    }
                };

                (
                    Some(Edge {
                        dist: Some(its.dist),
                        d: ray.d,
                    }),
                    Some(Vertex::Surface(SurfaceVertex {
                        its: its,
                        throughput: Color::one(),
                        sampled_bsdf: None,
                        rr_weight: 1.0,
                    })),
                )
            }
            Vertex::Surface(ref mut v) => {
                assert!(!sampler.is_none());
                let sampler = sampler.unwrap();

                v.sampled_bsdf = match v.its.mesh.bsdf.sample(
                    &v.its.uv,
                    &v.its.wi,
                    sampler.next2d(),
                ) {
                    Some(x) => Some(x),
                    None => return (None, None),
                };
                let sampled_bsdf = v.sampled_bsdf.as_ref().unwrap();

                // Update the throughput
                let mut new_throughput = v.throughput * sampled_bsdf.weight;
                if new_throughput.is_zero() {
                    return (None, None);
                }

                // Generate the new ray and do the intersection
                let d_out_global = v.its.frame.to_world(sampled_bsdf.d);
                let ray = Ray::new(v.its.p, d_out_global);
                let its = match scene.trace(&ray) {
                    Some(its) => its,
                    None => {
                        return (
                            Some(Edge {
                                dist: None,
                                d: d_out_global,
                            }),
                            None,
                        );
                    }
                };

                // Check RR
                let rr_weight = new_throughput.channel_max().min(0.95);
                if rr_weight < sampler.next() {
                    return (
                        Some(Edge {
                            dist: Some(its.dist),
                            d: d_out_global,
                        }),
                        None,
                    );
                }
                new_throughput /= rr_weight;

                (
                    Some(Edge {
                        dist: Some(its.dist),
                        d: d_out_global,
                    }),
                    Some(Vertex::Surface(SurfaceVertex {
                        its,
                        throughput: new_throughput,
                        sampled_bsdf: None,
                        rr_weight,
                    })),
                )
            }
        }
    }
}
