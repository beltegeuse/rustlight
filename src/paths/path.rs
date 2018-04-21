use cgmath::*;
use paths::vertex::*;
use samplers::*;
use scene::*;
use structure::*;

pub struct Path<'a> {
    pub vertices: Vec<Vertex<'a>>,
    pub edges: Vec<Edge>,
}

impl<'a> Path<'a> {
    pub fn from_sensor<S: Sampler>(
        (ix, iy): (u32, u32),
        scene: &'a Scene,
        sampler: &mut S,
        max_depth: Option<u32>,
    ) -> Option<Path<'a>> {
        let pix = Point2::new(ix as f32 + sampler.next(), iy as f32 + sampler.next());
        let mut vertices = vec![Vertex::new_sensor_vertex(pix, scene.camera.param.pos)];
        let mut edges: Vec<Edge> = vec![];

        let mut depth = 1;
        while max_depth.map_or(true, |max| depth < max) {
            match vertices
                .last_mut()
                .unwrap()
                .generate_next(scene, Some(sampler))
            {
                (Some(edge), Some(vertex)) => {
                    edges.push(edge);
                    vertices.push(vertex);
                }
                (Some(edge), None) => {
                    // This case model a path where we was able to generate a direction
                    // But somehow, not able to generate a intersection point, because:
                    //  - no geometry have been intersected
                    //  - russian roulette kill the path
                    edges.push(edge);
                    return Some(Path { vertices, edges });
                }
                _ => {
                    // Kill for a lot of reason ...
                    return Some(Path { vertices, edges });
                }
            }
            depth += 1;
        }

        Some(Path { vertices, edges })
    }

    pub fn get_img_position(&self) -> Point2<f32> {
        match &self.vertices[0] {
            &Vertex::Sensor(ref v) => v.uv,
            _ => panic!("Impossible to gather the base path image position"),
        }
    }
}
