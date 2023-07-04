use std::cmp::Ordering;
use std::mem::swap;

use crate::constants::EPSILON;
use crate::scene::Scene;
use crate::structure::*;
use cgmath::*;

pub trait Acceleration: Sync + Send {
    fn trace(&self, ray: &Ray) -> Option<Intersection>;
    fn visible(&self, p0: &Point3<f32>, p1: &Point3<f32>) -> bool;
}

pub struct NaiveAcceleration<'scene> {
    pub scene: &'scene Scene,
}
impl<'scene> NaiveAcceleration<'scene> {
    pub fn new(scene: &'scene Scene) -> NaiveAcceleration<'scene> {
        NaiveAcceleration { scene }
    }
}
impl<'a> Acceleration for NaiveAcceleration<'a> {
    fn trace(&self, ray: &Ray) -> Option<Intersection> {
        let mut its = IntersectionUV {
            t: std::f32::MAX,
            p: Point3::new(0.0, 0.0, 0.0),
            n: Vector3::new(0.0, 0.0, 0.0),
            u: 0.0,
            v: 0.0,
        };
        let (mut id_m, mut id_t) = (0, 0);

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
            let mesh = &self.scene.meshes[id_m];
            Some(Intersection::fill_intersection(
                mesh, id_t, its.u, its.v, ray, its.n, its.t, its.p,
            ))
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
            v: 0.0,
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

pub struct BVHNode {
    aabb: AABB,
    info: usize,
    count: usize,
}
impl BVHNode {
    pub fn is_leaf(&self) -> bool {
        self.count != 0
    }
}

#[derive(Clone)]
pub struct TriRef {
    id_mesh: usize,
    id_tri: usize,
}

pub struct CachedAABB {
    aabb: AABB,
    info: TriRef,
}

pub struct BVHAccel<'scene> {
    primitives: Vec<TriRef>,
    nodes: Vec<BVHNode>,
    scene: &'scene Scene,
}

fn compute_aabb(aabbs: &Vec<CachedAABB>, start: usize, count: usize) -> AABB {
    let mut aabb = AABB::default();
    for i in 0..count {
        aabb = aabb.union_aabb(&aabbs[i + start].aabb);
    }
    aabb
}

fn subdivide_node(
    id_node: usize,
    aabbs: &mut Vec<CachedAABB>,
    nodes: &mut Vec<BVHNode>,
    depth: usize,
) {
    // Decision to transform the node or not
    let (nb_prim, first_prim) = {
        if nodes[id_node].count <= 2 {
            return; // Stop here
        }

        let nb_prim = nodes[id_node].count;
        let first_prim = nodes[id_node].info;
        nodes[id_node].count = 0; // Transform to node
        nodes[id_node].info = nodes.len(); // Ptr to the next node that we will create
        (nb_prim, first_prim)
    };

    let offset = {
        // Use SAH Sweep
        let mut best_pos = 0;
        let mut best_cost = std::f32::INFINITY;
        let mut best_axis = 3;
        {
            let mut scores = vec![0.0; nb_prim - 1];
            for o in 0..3 {
                aabbs[first_prim..first_prim + nb_prim].sort_by(|a, b| -> Ordering {
                    if a.aabb.center()[o] < b.aabb.center()[o] {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    }
                });
                let mut tmp = AABB::default();
                for id in 0..nb_prim - 1 {
                    let id_left = nb_prim - id - 1;
                    tmp = tmp.union_aabb(&aabbs[id_left + first_prim].aabb);
                    scores[id_left - 1] = tmp.surface_area() * (id + 1) as f32;
                }
                tmp = AABB::default();
                for id in 0..nb_prim - 1 {
                    tmp = tmp.union_aabb(&aabbs[id + first_prim].aabb);
                    scores[id] += tmp.surface_area() * (id + 1) as f32;
                    if scores[id] < best_cost {
                        best_cost = scores[id];
                        best_axis = o;
                        best_pos = id + 1;
                    }
                }
            }
        }
        // Sort on the selected axis
        aabbs[first_prim..first_prim + nb_prim].sort_by(|a, b| -> Ordering {
            if a.aabb.center()[best_axis] < b.aabb.center()[best_axis] {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        });
        if best_pos == nb_prim || best_pos == 0 {
            ((nb_prim as f32 * 0.5) as usize).max(1)
        } else {
            best_pos
        }
    };

    let left = BVHNode {
        aabb: compute_aabb(aabbs, first_prim, offset),
        info: first_prim,
        count: offset,
    };
    let right = BVHNode {
        aabb: compute_aabb(aabbs, first_prim + offset, nb_prim - offset),
        info: first_prim + offset,
        count: nb_prim - offset,
    };
    let id_left = nodes.len();
    nodes.push(left);
    nodes.push(right);

    // Do the recursive calls
    subdivide_node(id_left, aabbs, nodes, depth + 1);
    subdivide_node(id_left + 1, aabbs, nodes, depth + 1);
}

impl<'scene> BVHAccel<'scene> {
    pub fn new(scene: &'scene Scene) -> BVHAccel<'scene> {
        // Compute AABB cached
        let mut root_aabb = AABB::default();
        let mut cached_aabbs = Vec::new();
        for m in 0..scene.meshes.len() {
            let mesh = &scene.meshes[m];
            for i in 0..mesh.indices.len() {
                cached_aabbs.push(CachedAABB {
                    aabb: mesh.compute_aabb_tri(i),
                    info: TriRef {
                        id_mesh: m,
                        id_tri: i,
                    },
                });
                root_aabb = root_aabb.union_aabb(&cached_aabbs.last().unwrap().aabb)
            }
        }

        // Construct the root node
        let mut nodes = Vec::new();
        let root = BVHNode {
            aabb: root_aabb,
            info: 0,
            count: cached_aabbs.len(),
        };
        nodes.push(root);

        // Subdivide recursively the root node
        subdivide_node(0, &mut cached_aabbs, &mut nodes, 0);

        // Generate list of primitives
        let primitives = cached_aabbs.iter().map(|c| c.info.clone()).collect();
        BVHAccel {
            primitives,
            nodes,
            scene,
        }
    }
}

impl<'a> BVHAccel<'a> {
    pub fn intersect(&self, id_node: usize, ray: &Ray, its: &mut IntersectionUV) -> Option<TriRef> {
        let node = &self.nodes[id_node];
        if node.is_leaf() {
            let mut res = None;
            for k in 0..node.count {
                let e = &self.primitives[node.info + k];
                if self.scene.meshes[e.id_mesh].intersection_tri(e.id_tri, &ray.o, &ray.d, its) {
                    res = Some(e.clone());
                }
            }
            return res;
        } else {
            // Get the nodes
            let mut id1 = node.info;
            let mut id2 = node.info + 1;
            // Check AABB and distance
            let d1 = self.nodes[id1].aabb.intersect(ray);
            let d2 = self.nodes[id2].aabb.intersect(ray);
            let mut d1 = if let Some(v) = d1 {
                v
            } else {
                std::f32::INFINITY
            };
            let mut d2 = if let Some(v) = d2 {
                v
            } else {
                std::f32::INFINITY
            };
            // Found the cloest one
            if d1 > d2 {
                swap(&mut d1, &mut d2);
                swap(&mut id1, &mut id2);
            }
            let mut res = None;
            if d1 < its.t {
                res = self.intersect(id1, ray, its);
            }
            if d2 < its.t {
                let res2 = self.intersect(id2, ray, its);
                if res2.is_some() {
                    res = res2;
                }
            }
            res
        }
    }
}

impl<'a> Acceleration for BVHAccel<'a> {
    fn trace(&self, ray: &Ray) -> Option<Intersection> {
        if self.nodes[0].aabb.intersect(ray).is_none() {
            return None;
        }

        let mut its = IntersectionUV {
            t: std::f32::MAX,
            p: Point3::new(0.0, 0.0, 0.0),
            n: Vector3::new(0.0, 0.0, 0.0),
            u: 0.0,
            v: 0.0,
        };
        let res = self.intersect(0, ray, &mut its);

        if its.t == std::f32::MAX {
            None
        } else {
            let res = res.unwrap();
            let mesh = &self.scene.meshes[res.id_mesh];
            Some(Intersection::fill_intersection(
                mesh, res.id_tri, its.u, its.v, ray, its.n, its.t, its.p,
            ))
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
            v: 0.0,
        };
        let ray = Ray {
            o: *p0,
            d,
            tnear: EPSILON,
            tfar: length * (1.0 - SHADOW_EPS),
        };

        // TODO: Unecessary?
        if self.nodes[0].aabb.intersect(&ray).is_none() {
            return false;
        }

        return self.intersect(0, &ray, &mut its).is_none();
    }
}

#[cfg(feature = "embree")]
pub struct EmbreeAcceleration<'scene, 'embree> {
    pub scene: &'scene Scene,
    pub embree_scene_commited: embree::CommittedScene<'embree>,
}
#[cfg(feature = "embree")]
impl<'scene, 'embree> EmbreeAcceleration<'scene, 'embree> {
    pub fn new(
        scene: &'scene Scene,
        embree_scene: &'embree embree::Scene,
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
        let mut intersection_ctx = embree::IntersectContext::incoherent();
        let embree_ray = embree::Ray::segment(
            Vector3::new(ray.o.x, ray.o.y, ray.o.z),
            ray.d,
            ray.tnear,
            ray.tfar,
        );
        let mut ray_hit = embree::RayHit::new(embree_ray);
        self.embree_scene_commited
            .intersect(&mut intersection_ctx, &mut ray_hit);
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

            let mesh = &self.scene.meshes[ray_hit.hit.geomID as usize];
            Some(Intersection::fill_intersection(
                mesh,
                ray_hit.hit.primID as usize,
                ray_hit.hit.u,
                ray_hit.hit.v,
                ray,
                n_g,
                ray_hit.ray.tfar,
                p,
            ))
        } else {
            None
        }
    }
    fn visible(&self, p0: &Point3<f32>, p1: &Point3<f32>) -> bool {
        let mut intersection_ctx = embree::IntersectContext::coherent();
        let mut d = p1 - p0;
        let length = d.magnitude();
        d /= length;
        // TODO: Do correct self intersection tests...
        let mut embree_ray =
            embree::Ray::segment(Vector3::new(p0.x, p0.y, p0.z), d, 0.0001, length - 0.0001);
        self.embree_scene_commited
            .occluded(&mut intersection_ctx, &mut embree_ray);
        embree_ray.tfar != std::f32::NEG_INFINITY
    }
}

/**
 * General BVH
 * Usefull for accelerating general intersections (such planes)
 */

#[derive(Debug)]
struct BVHNodeGeneral {
    pub aabb: AABB,
    pub first: usize,
    pub count: usize,
    pub left: Option<usize>,
    pub right: Option<usize>,
}

impl BVHNodeGeneral {
    pub fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }
}

pub struct BHVAccel<D, T: BVHElement<D>> {
    pub elements: Vec<T>,
    nodes: Vec<BVHNodeGeneral>,
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
            self.nodes.push(BVHNodeGeneral {
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
            self.nodes.push(BVHNodeGeneral {
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

    pub fn gather(&self, r: &Ray) -> Vec<(D, usize)> {
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
            let t_aabb = n.aabb.intersect(r);
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
