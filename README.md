<h1>
Rustlight <img src="http://beltegeuse.s3-website-ap-northeast-1.amazonaws.com/rustlight/logo.png" width="96"> 
</h1>

[![Build Status](https://travis-ci.org/beltegeuse/rustlight.svg?branch=master)](https://travis-ci.org/beltegeuse/rustlight)

Physically-based rendering engine implemented with **Rust**.

## How to use it

```
$ cargo run --release -- -h
rustlight 0.1.0
Adrien Gruson <adrien.gruson@gmail.com>
A Rusty Light Transport simulation program

USAGE:
    rustlight [FLAGS] [OPTIONS] <scene> [SUBCOMMAND]

FLAGS:
    -d               debug output
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -a <average>            average several pass of the integrator with a time limit ('inf' is possible)
    -s <image_scale>        image scaling factor [default: 1.0]
    -n <nbsamples>          integration technique
    -t <nbthreads>          number of thread for the computation [default: auto]
    -o <output>             output image file

ARGS:
    <scene>    JSON file description

SUBCOMMANDS:
    ao                ambiant occlusion
    direct            direct lighting
    gradient-path     gradient path tracing
    help              Prints this message or the help of the given subcommand(s)
    light-explicit    light tracing with explict light path construction
    path              path tracing
    path-explicit     path tracing with explict light path construction
    pssmlt            path tracing with MCMC sampling
    vpl               brute force virtual point light integrator
```

For example, to use path tracing using 128 spp:
```
$ cargo run --release -- -n 128 -o path.pfm ./data/cbox.json path
```

## Dependencies

Optionals : 

- [image](https://github.com/image-rs/image) : load and save LDR images
- [openexr](https://github.com/cessen/openexr-rs) : load and save EXR images
- [embree-rs](https://github.com/Twinklebear/embree-rs) : fast primitive/ray intersection (* not yet optional)
- [pbrt_rs](https://github.com/beltegeuse/pbrt_rs) : read PBRT files 

## Features

For now, these are the following features implemented:
- Integrators: 
    * Ambiant occlusion
    * Direct with MIS
    * Path-tracing with NEE
    * Gradient-path tracing [1]
    * Primary-sample space MLT [2]
- Explicit Integrators: Uses a graph to represent the light transport
    * Path tracing with NEE (*2 slower~ than non-explicit one)
    * Light tracing
    * Virtual Point Light
- Filtering: 
    * Image-space control variate with uniform and variance-based weights [3]
- Materials: 
    * Diffuse
    * Phong lobe
    * Specular
    * A subset of PBRT materials (imported from [rs_pbrt](https://github.com/wahn/rs_pbrt))
- Emitters: 
    * Multiple tri-mesh lights support

## Rendering

![Cornel Box gradient-domain pt](http://beltegeuse.s3-website-ap-northeast-1.amazonaws.com/rustlight/pbrt_rs.png)

## Roadmap

Rendering algorithms for path-tracing:

- use the explict layout to do implement gradient-domain path tracing
- fixing gradient-domain path tracing: seems to have wrong gradient when the light source is not visible from the base path
- gradient-domain path reuse

Other rendering features:

- Materials: glass, microfacet with Beckert distribution.
- Emitters: Environmental and point lights
- Scene format support: PBRT

## Inspirations

This code has been inspired from several repositories:

- rs_pbrt project: https://github.com/wahn/rs_pbrt
- the blog post from Brook Heisler: https://bheisler.github.io/post/writing-raytracer-in-rust-part-1/
- tray_rust project: https://github.com/Twinklebear/tray_rust
- mitsuba: https://github.com/mitsuba-renderer/mitsuba

## References
[1] Kettunen, Markus, et al. "Gradient-domain path tracing." ACM Transactions on Graphics (TOG) (2015) \
[2] Kelemen, Csaba, et al. "A simple and robust mutation strategy for the metropolis light transport algorithm." Computer Graphics Forum (2002) \
[3] Rousselle, Fabrice, et al. "Image-space control variates for rendering." ACM Transactions on Graphics (TOG) (2016) 
