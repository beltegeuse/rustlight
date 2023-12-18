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
#[cfg(feature = "embree")]
extern crate embree;
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
// For loading the obj files
extern crate tobj;
// For print a progress bar
#[cfg(feature = "progress-bar")]
extern crate pbr;
// For loading other type of scene format
#[cfg(feature = "mitsuba")]
extern crate mitsuba_rs;
#[cfg(feature = "pbrt")]
extern crate pbrt_rs;
// For building hashmap in build time
#[macro_use]
extern crate lazy_static;
// For making assertion over floats
#[macro_use]
extern crate assert_approx_eq;
// For flag bit implementation (C++)
#[macro_use]
extern crate bitflags;

mod constants {
    pub const EPSILON: f32 = 0.0001;
    pub const ONE_MINUS_EPSILON: f32 = 0.9999999403953552;
}

pub trait Scale<T> {
    fn scale(&mut self, v: T);
}

fn clamp<T: PartialOrd>(v: T, min: T, max: T) -> T {
    if v < min {
        min
    } else if v > max {
        max
    } else {
        v
    }
}

// all the modules
pub mod accel;
pub mod bsdfs;
pub mod camera;
pub mod color;
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
