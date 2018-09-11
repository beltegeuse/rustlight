#![cfg_attr(feature = "clippy", feature(plugin))]
#![cfg_attr(feature = "clippy", plugin(clippy))]

// For getting low machine information
extern crate byteorder;
// For the vector op
extern crate cgmath;
// For fast intersection
extern crate embree_rs;
// For the image (LDR) export and loading
extern crate image;
// For the image (HDR) export and loading
extern crate openexr;
// For logging propose
#[macro_use]
extern crate log;
// For the random number generator
extern crate rand;
// For easy parallelism
extern crate rayon;
// For serialization support
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;
// For loading the obj files
extern crate tobj;
// For print a progress bar
extern crate pbr;

mod constants {
    pub const EPSILON: f32 = 0.0001;
}

pub trait Scale<T> {
    fn scale(&mut self, v: T);
}

// all the modules
pub mod bsdfs;
pub mod camera;
pub mod geometry;
pub mod integrators;
pub mod math;
pub mod paths;
pub mod samplers;
pub mod scene;
pub mod structure;
pub mod tools;
