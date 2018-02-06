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


pub mod rustlight;