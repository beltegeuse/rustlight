#![cfg_attr(feature = "clippy", feature(plugin))]
#![cfg_attr(feature = "clippy", plugin(clippy))]

// For the vector op
extern crate cgmath;
// For fast intersection
extern crate embree_rs;
// For the image (LDR) export
extern crate image;
// For logging propose
#[macro_use]
extern crate log;
// For the random number generator
extern crate rand;
// For easy parallelism
extern crate rayon;
extern crate serde;
// For serialization support
#[macro_use]
extern crate serde_derive;
extern crate serde_json;
// For loading the obj files
extern crate tobj;

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
pub mod path;
pub mod samplers;
pub mod scene;
pub mod structure;
pub mod tools;
