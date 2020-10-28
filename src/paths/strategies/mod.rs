use crate::accel::*;
use crate::paths::path::*;
use crate::paths::vertex::*;
use crate::samplers::*;
use crate::scene::*;
use crate::structure::*;
use crate::volume::*;
use std;
use std::mem;

pub trait SamplingStrategy {
    fn sample<'scene>(
        &self,
        path: &mut Path<'scene>,
        vertex_id: VertexID,
        accel: &'scene dyn Acceleration,
        scene: &'scene Scene,
        throughput: Color,
        sampler: &mut dyn Sampler,
        medium: Option<&HomogenousVolume>,
        id_strategy: usize,
        depth: u32,
    ) -> Option<(VertexID, Color)>;

    // All PDF have to be inside the same domain
    fn pdf<'scene>(
        &self,
        path: &Path<'scene>,
        scene: &'scene Scene,
        vertex_id: VertexID,
        edge_id: EdgeID,
    ) -> Option<f32>;
}

pub fn generate<'scene, T: Technique>(
    path: &mut Path<'scene>,
    root: VertexID,
    accel: &'scene dyn Acceleration,
    scene: &'scene Scene,
    sampler: &mut dyn Sampler,
    technique: &mut T,
) {
    let mut curr = vec![(root, Color::one())];
    let mut next = vec![];
    let mut depth = 1;
    while !curr.is_empty() {
        // Do a wavefront processing of the different vertices
        next.clear();
        for (curr_vertex_id, throughput) in &curr {
            // For all the sampling techniques
            // This is the continue if we want to continue or not
            // For example, we might want to not push the vertex if we have reach the depth limit
            if technique.expand(path.vertex(*curr_vertex_id), depth) {
                for (id_sampling, sampling) in technique
                    .strategies(path.vertex(*curr_vertex_id))
                    .iter()
                    .enumerate()
                {
                    // If we want to continue the tracing toward this direction
                    if let Some((new_vertex, new_throughput)) = sampling.sample(
                        path,
                        *curr_vertex_id,
                        accel,
                        scene,
                        *throughput,
                        sampler,
                        scene.volume.as_ref(), // TODO: For now volume is global
                        id_sampling,
                        depth,
                    ) {
                        next.push((new_vertex, new_throughput));
                    }
                }
            }
        }
        // Flip-flap buffer
        mem::swap(&mut curr, &mut next);
        depth += 1;
    }
}

pub trait Technique {
    fn strategies(&self, vertex: &Vertex) -> &Vec<Box<dyn SamplingStrategy>>;
    fn expand(&self, vertex: &Vertex, depth: u32) -> bool;
}

pub mod directional;
pub mod emitters;
pub mod naive;
