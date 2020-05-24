#![cfg_attr(feature = "clippy", feature(plugin))]
#![cfg_attr(feature = "clippy", plugin(clippy))]
#![allow(dead_code)]
#![allow(clippy::float_cmp)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_arguments)]

// For getting low machine information
extern crate byteorder;
// For the vector op
extern crate cgmath;
// For fast intersection
extern crate embree_rs;
// For the image (LDR) export and loading
#[cfg(feature = "image")]
extern crate image;
// For the image (HDR) export and loading
#[cfg(feature = "openexr")]
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
#[cfg(feature = "progress-bar")]
extern crate pbr;
// For loading other type of scene format
#[cfg(feature = "pbrt")]
extern crate pbrt_rs;

mod constants {
    pub const EPSILON: f32 = 0.0001;
}

pub trait Scale<T> {
    fn scale(&mut self, v: T);
}

// all the modules
pub mod accel;
pub mod bsdfs;
pub mod camera;
pub mod emitter;
pub mod geometry;
pub mod integrators;
pub mod math;
pub mod paths;
pub mod samplers;
pub mod scene;
pub mod scene_loader;
pub mod structure;
pub mod tools;
pub mod volume;
