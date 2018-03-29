#![cfg_attr(feature = "clippy", feature(plugin))]
#![cfg_attr(feature = "clippy", plugin(clippy))]

// For the vector op
extern crate cgmath;
// For fast intersection
extern crate embree_rs;
// For the image (LDR) export
extern crate image;
// For print a progress bar
extern crate pbr;
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
// For logging propose
#[macro_use]
extern crate log;

mod constants {
    pub const EPSILON: f32 = 0.0001;
}

use std::ops::AddAssign;

pub trait Scale<T> {
    fn scale(&mut self, v: T);
}

pub trait BitmapTrait: Default + AddAssign + Scale<f32> + Clone {}

// all the modules
pub mod structure;
pub mod sampler;
pub mod material;
pub mod geometry;
pub mod camera;
pub mod scene;
pub mod integrator;
pub mod math;
pub mod path;
mod tools; // private module