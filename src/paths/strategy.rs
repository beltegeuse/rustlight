use crate::accel::*;
use crate::emitter::*;
use crate::paths::path::*;
use crate::paths::vertex::*;
use crate::samplers::*;
use crate::scene::*;
use crate::structure::*;
use crate::volume::*;
use std;
use std::mem;

pub trait SamplingStrategy {
    fn sample<'scene, 'emitter>(
        &self,
        path: &mut Path<'scene, 'emitter>,
        vertex_id: VertexID,
        accel: &'scene dyn Acceleration,
        scene: &'scene Scene,
        emitters: &'emitter EmitterSampler,
        throughput: Color,
        sampler: &mut dyn Sampler,
        medium: Option<&HomogenousVolume>,
        id_strategy: usize,
    ) -> Option<(VertexID, Color)>;

    // All PDF have to be inside the same domain
    fn pdf<'scene, 'emitter>(
        &self,
        path: &Path<'scene, 'emitter>,
        scene: &'scene Scene,
        emitters: &'emitter EmitterSampler,
        vertex_id: VertexID,
        edge_id: EdgeID,
    ) -> Option<f32>;
}

pub fn generate<'scene, 'emitter, T: Technique>(
    path: &mut Path<'scene, 'emitter>,
    accel: &'scene dyn Acceleration,
    scene: &'scene Scene,
    emitters: &'emitter EmitterSampler,
    sampler: &mut dyn Sampler,
    technique: &mut T,
) -> Vec<(VertexID, Color)> {
    let root = technique.init(path, accel, scene, sampler, emitters);
    let mut curr = root.clone();
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
                        emitters,
                        *throughput,
                        sampler,
                        scene.volume.as_ref(), // TODO: For now volume is global
                        id_sampling,
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

    root
}

pub trait Technique {
    fn init<'scene, 'emitter>(
        &mut self,
        path: &mut Path<'scene, 'emitter>,
        accel: &'scene dyn Acceleration,
        scene: &'scene Scene,
        sampler: &mut dyn Sampler,
        emitters: &'emitter EmitterSampler,
    ) -> Vec<(VertexID, Color)>;
    fn strategies(&self, vertex: &Vertex) -> &Vec<Box<dyn SamplingStrategy>>;
    fn expand(&self, vertex: &Vertex, depth: u32) -> bool;
}
