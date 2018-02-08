// For the vector op
extern crate cgmath;
// For easy parallelism
extern crate rayon;
// For the image (LDR) export
extern crate image;
// For serialization support
#[macro_use] extern crate serde_derive;
extern crate serde;
extern crate serde_json;
// For fast intersection
extern crate embree;
// For the random number generator
extern crate rand;
// For loading the obj files
extern crate tobj;

// all the modules
pub mod structure;
pub mod sampler;
pub mod material;
pub mod geometry;
pub mod camera;
pub mod scene;
pub mod integrator;
pub mod math;