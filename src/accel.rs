use crate::structure::*;
use crate::scene::Scene;
use cgmath::*;

#[derive(Debug)]
struct BVHNode {
    pub aabb: AABB,
    pub first: usize,
    pub count: usize,
    pub left: Option<usize>,
    pub right: Option<usize>,
}

impl BVHNode {
    pub fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }
}

pub struct BHVAccel<D, T: BVHElement<D>> {
    pub elements: Vec<T>,
    nodes: Vec<BVHNode>,
    pub root: Option<usize>, // Root node
    phantom: std::marker::PhantomData<D>,
}

pub trait BVHElement<D> {
    // Used to build AABB hierachy
    fn aabb(&self) -> AABB;
    // Used to construct AABB (by sorting elements)
    fn position(&self) -> Point3<f32>;
    // Used when collecting the different objects
    fn intersection(&self, r: &Ray) -> Option<D>;
}

// Implementation from (C++): https://github.com/shiinamiyuki/minpt/blob/master/minpt.cpp
impl<D, T: BVHElement<D>> BHVAccel<D, T> {
    // Internal build function
    // return the node ID (allocate node on the fly)
    fn build(&mut self, begin: usize, end: usize, depth: u32) -> Option<usize> {
        if end == begin {
            return None;
        }

        // TODO: Not very optimized ...
        let mut aabb = AABB::default();
        for i in begin..end {
            aabb = aabb.union_aabb(&self.elements[i].aabb());
        }

        // If the tree is too deep or not enough element
        // force to make a leaf
        // depth >= 20
        if end - begin <= 4 {
            // dbg!(aabb.size());
            self.nodes.push(BVHNode {
                aabb,
                first: begin,
                count: end - begin,
                left: None,
                right: None,
            });
            Some(self.nodes.len() - 1)
        } else {
            // For now cut on the biggest axis
            let aabb_size = aabb.size();
            let axis = if aabb_size.x > aabb_size.y {
                if aabb_size.x > aabb_size.z {
                    0
                } else {
                    2
                }
            } else if aabb_size.y > aabb_size.z {
                1
            } else {
                2
            };

            // Split based on largest axis (split inside the middle for now)
            // or split between triangles
            // TODO: Implements SAH
            // let split = (aabb.p_max[axis] + aabb.p_min[axis]) / 2.0;
            // let split_id = self.triangles[begin..end].iter_mut().partition_in_place(|t| t.middle()[axis] < split ) + begin;

            self.elements[begin..end].sort_unstable_by(|t1, t2| {
                t1.position()[axis]
                    .partial_cmp(&t2.position()[axis])
                    .unwrap()
            });
            let split_id = (begin + end) / 2;

            // TODO: Make better
            let left = self.build(begin, split_id, depth + 1);
            let right = self.build(split_id, end, depth + 1);
            self.nodes.push(BVHNode {
                aabb,
                first: 0, // TODO: Make the node invalid
                count: 0,
                left,
                right,
            });

            Some(self.nodes.len() - 1)
        }
    }

    pub fn create(elements: Vec<T>) -> BHVAccel<D, T> {
        let mut accel = BHVAccel {
            elements,
            nodes: Vec::new(),
            root: None,
            phantom: std::marker::PhantomData,
        };
        accel.root = accel.build(0, accel.elements.len(), 0);
        match accel.root {
            None => warn!("BVH is empty!"),
            Some(ref v) => {
                info!("BVH stats: ");
                info!(" - Number of elements: {}", accel.elements.len());
                info!(" - Number of nodes: {}", accel.nodes.len());
                info!(" - AABB size root: {:?}", accel.nodes[*v].aabb.size());
            }
        };
        accel
    }

    pub fn gather(&self, r: Ray) -> Vec<(D, usize)> {
        let mut res = vec![];
        if self.root.is_none() {
            return res;
        }

        // Indices of nodes
        let mut stack: Vec<usize> = Vec::new();
        stack.reserve(100); // In case
        stack.push(self.root.unwrap());

        while let Some(curr_id) = stack.pop() {
            let n = &self.nodes[curr_id];
            let t_aabb = n.aabb.intersect(&r);
            match (n.is_leaf(), t_aabb) {
                (_, None) => {
                    // Nothing to do as we miss the node
                }
                (true, Some(_t)) => {
                    for i in n.first..(n.first + n.count) {
                        if let Some(d) = self.elements[i].intersection(&r) {
                            res.push((d, i));
                        }
                    }
                }
                (false, Some(_t)) => {
                    if let Some(left_id) = n.left {
                        stack.push(left_id);
                    }
                    if let Some(right_id) = n.right {
                        stack.push(right_id);
                    }
                }
            }
        }
        res
    }
}


pub trait Acceleration: Sync + Send {
    fn trace(&self, ray: &Ray) -> Option<Intersection>;
    fn visible(&self, p0: &Point3<f32>, p1: &Point3<f32>) -> bool;
}

pub struct NaiveAcceleration<'scene> {
    pub scene: &'scene Scene
}
impl<'scene> NaiveAcceleration<'scene> {
    pub fn new(scene: &'scene Scene) -> NaiveAcceleration<'scene> {
        NaiveAcceleration {
            scene
        }
    }
}
impl<'a> Acceleration for NaiveAcceleration<'a> {    
    fn trace(&self, ray: &Ray) -> Option<Intersection> {
        let mut its = IntersectionUV {
            t: std::f32::MAX,
            p: Point3::new(0.0, 0.0, 0.0),
            n: Vector3::new(0.0, 0.0, 0.0),
            u: 0.0,
            v: 0.0
        };
        let (mut id_m, mut id_t) = (0,0);

        for m in 0..self.scene.meshes.len() {
            let mesh = &self.scene.meshes[m];
            for i in 0..mesh.indices.len() {
                if mesh.intersection_tri(i, &ray.o, &ray.d, &mut its) {
                    id_m = m;
                    id_t = i; 
                }
            }
        }

        if its.t == std::f32::MAX {
            None
        } else {
            Some(Intersection::fill_intersection(id_m, id_t, self.scene, 
                its.u, its.v, ray, its.n, its.t, its.p))
        }
    }
    fn visible(&self, p0: &Point3<f32>, p1: &Point3<f32>) -> bool {
        const SHADOW_EPS: f32 = 0.00001;
        // Compute ray dir
        let mut d = p1 - p0;
        let length = d.magnitude();
        d /= length;

        let mut its = IntersectionUV {
            t: length * (1.0 - SHADOW_EPS),
            p: Point3::new(0.0, 0.0, 0.0),
            n: Vector3::new(0.0, 0.0, 0.0),
            u: 0.0,
            v: 0.0
        };
        
        for m in 0..self.scene.meshes.len() {
            let mesh = &self.scene.meshes[m];
            for i in 0..mesh.indices.len() {
                if mesh.intersection_tri(i, &p0, &d, &mut its) {
                    return false;
                }
            }
        }
        return true; 
    }
}

#[cfg(feature = "embree")]
pub struct EmbreeAcceleration<'scene, 'embree> {
    pub scene: &'scene Scene,
    pub embree_scene_commited: embree_rs::CommittedScene<'embree>,
}
#[cfg(feature = "embree")]
impl<'scene, 'embree> EmbreeAcceleration<'scene, 'embree> {
    pub fn new(
        scene: &'scene Scene,
        embree_scene: &'embree embree_rs::Scene,
    ) -> EmbreeAcceleration<'scene, 'embree> {
        EmbreeAcceleration {
            scene,
            embree_scene_commited: embree_scene.commit(),
        }
    }
}

#[cfg(feature = "embree")]
impl<'scene, 'embree> Acceleration for EmbreeAcceleration<'scene, 'embree> {
    fn trace(&self, ray: &Ray) -> Option<Intersection> {
        let mut intersection_ctx = embree_rs::IntersectContext::coherent();
        let embree_ray = embree_rs::Ray::segment(
            Vector3::new(ray.o.x, ray.o.y, ray.o.z),
            ray.d,
            ray.tnear,
            ray.tfar,
        );
        let mut ray_hit = embree_rs::RayHit::new(embree_ray);
        self.embree_scene_commited.intersect(&mut intersection_ctx, &mut ray_hit);
        if ray_hit.hit.hit() {
            let mut n_g = Vector3::new(ray_hit.hit.Ng_x, ray_hit.hit.Ng_y, ray_hit.hit.Ng_z);
            let n_g_dot = n_g.dot(n_g);
            if n_g_dot != 1.0 {
                n_g /= n_g_dot.sqrt();
            }
            let p = Point3::new(
                ray_hit.ray.org_x + ray_hit.ray.tfar * ray_hit.ray.dir_x,
                ray_hit.ray.org_y + ray_hit.ray.tfar * ray_hit.ray.dir_y,
                ray_hit.ray.org_z + ray_hit.ray.tfar * ray_hit.ray.dir_z,
            );
            Some(Intersection::fill_intersection(ray_hit.hit.geomID as usize, 
                ray_hit.hit.primID as usize, self.scene, 
                ray_hit.hit.u, ray_hit.hit.v, ray, n_g, ray_hit.ray.tfar, p))
        } else {
            None
        }
    }
    fn visible(&self, p0: &Point3<f32>, p1: &Point3<f32>) -> bool {
        let mut intersection_ctx = embree_rs::IntersectContext::coherent();
        let mut d = p1 - p0;
        let length = d.magnitude();
        d /= length;
        let mut embree_ray =
            embree_rs::Ray::segment(Vector3::new(p0.x, p0.y, p0.z), d, 0.00001, length - 0.00001);
        self.embree_scene_commited
            .occluded(&mut intersection_ctx, &mut embree_ray);
        embree_ray.tfar != std::f32::NEG_INFINITY
    }
}