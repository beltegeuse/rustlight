use crate::structure::*;
use cgmath::Point3;

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

pub struct BHVAccel<T: BVHElement> {
    pub elements: Vec<T>,
    nodes: Vec<BVHNode>,
    pub root: Option<usize>, // Root node
}

pub trait BVHElement {
    // Used to build AABB hierachy
    fn aabb(&self) -> AABB;
    // Used to construct AABB (by sorting elements)
    fn position(&self) -> Point3<f32>;
    // Used when collecting the different objects
    fn intersection(&self, r: &Ray) -> Option<f32>;
}

// Implementation from (C++): https://github.com/shiinamiyuki/minpt/blob/master/minpt.cpp
impl<T: BVHElement> BHVAccel<T> {
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
            } else {
                if aabb_size.y > aabb_size.z {
                    1
                } else {
                    2
                }
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

    pub fn create(elements: Vec<T>) -> BHVAccel<T> {
        let mut accel = BHVAccel {
            elements,
            nodes: Vec::new(),
            root: None,
        };
        accel.root = accel.build(0, accel.elements.len(), 0);
        info!("BVH stats: ");
        info!(" - Number of elements: {}", accel.elements.len());
        info!(" - Number of nodes: {}", accel.nodes.len());
        info!(
            " - AABB size root: {:?}",
            accel.nodes[accel.root.unwrap()].aabb.size()
        );
        accel
    }

    pub fn gather(&self, r: Ray) -> Vec<(f32, usize)> {
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
