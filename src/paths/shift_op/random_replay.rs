use cgmath::*;
use paths::path::*;
use paths::shift_path::*;
use paths::vertex::*;
use samplers::*;
use scene::*;

pub struct ReplaySampler<'sampler, 'seq> {
    pub sampler: &'sampler mut Sampler,
    pub random: &'seq mut Vec<f32>,
    pub indice: usize,
}
impl<'sampler, 'seq> ReplaySampler<'sampler, 'seq> {
    fn generate(&mut self) -> f32 {
        assert!(self.indice <= self.random.len());
        if self.indice < self.random.len() {
            let v = self.indice;
            self.indice += 1;
            self.random[v]
        } else {
            let v = self.sampler.next();
            self.indice += 1;
            self.random.push(v);
            v
        }
    }
}
impl<'sampler, 'seq> Sampler for ReplaySampler<'sampler, 'seq> {
    fn next(&mut self) -> f32 {
        self.generate()
    }
    fn next2d(&mut self) -> Point2<f32> {
        let v1 = self.generate();
        let v2 = self.generate();
        Point2::new(v1, v2)
    }
}
pub struct ShiftRandomReplay {
    pub random_sequence: Vec<f32>,
}
impl Default for ShiftRandomReplay {
    fn default() -> Self {
        Self {
            random_sequence: vec![],
        }
    }
}
impl<'a> ShiftOp<'a> for ShiftRandomReplay {
    fn generate_base<S: Sampler>(
        &mut self,
        (ix, iy): (u32, u32),
        scene: &'a Scene,
        sampler: &mut S,
        max_depth: Option<u32>,
    ) -> Option<Path<'a>> {
        // Generate the base path
        self.random_sequence = vec![];
        let mut capture_sampler = ReplaySampler {
            sampler,
            random: &mut self.random_sequence,
            indice: 0,
        };
        let path = Path::from_sensor((ix, iy), scene, &mut capture_sampler, max_depth);
        path
    }

    fn shift<S: Sampler>(
        &mut self,
        base_path: &Path<'a>,
        shift_pix: Point2<f32>,
        scene: &'a Scene,
        sampler: &mut S,
        max_depth: Option<u32>,
    ) -> Option<ShiftPath<'a>> {
        let mut replay_sampler = ReplaySampler {
            sampler,
            random: &mut self.random_sequence,
            indice: 0,
        };
        // Generate the shift path
        let shift_path = Path::from_sensor(
            (shift_pix.x as u32, shift_pix.y as u32),
            scene,
            &mut replay_sampler,
            max_depth,
        );
        // Convert the shift path
        if shift_path.is_none() {
            return None;
        }
        let shift_path = shift_path.unwrap();
        let mut pdf_ratio = 1.0;
        Some(ShiftPath {
            vertices: shift_path
                .vertices
                .into_iter()
                .enumerate()
                .map(|(i, v)| {
                    match v {
                        Vertex::Sensor(v) => Some(ShiftVertex::Sensor(v)),
                        Vertex::Surface(v) => {
                            match base_path.vertices.get(i) {
                                None => None,
                                Some(&Vertex::Surface(ref main_next)) => {
                                    let current_pdf_ratio = pdf_ratio;
                                    if !main_next.sampled_bsdf.is_none()
                                        && !v.sampled_bsdf.is_none()
                                    {
                                        // FIXME: Check the measure
                                        pdf_ratio *=
                                            main_next.sampled_bsdf.as_ref().unwrap().pdf.value()
                                                / v.sampled_bsdf.unwrap().pdf.value();
                                    }
                                    Some(ShiftVertex::Surface(SurfaceVertexShift {
                                        its: v.its,
                                        throughput: v.throughput,
                                        pdf_ratio: current_pdf_ratio,
                                    }))
                                }
                                _ => panic!("Encounter wrong type"),
                            }
                        }
                    }
                })
                .filter(|v| !v.is_none())
                .map(|v| v.unwrap())
                .collect(),
            edges: shift_path.edges, // FIXME: Need to prune to have similar number of edges
        })
    }
}
