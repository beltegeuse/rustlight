<h1>
Rustlight <img src="http://beltegeuse.s3-website-ap-northeast-1.amazonaws.com/rustlight/logo.png" width="96"> 
</h1>

[![Build Status](https://travis-ci.org/beltegeuse/rustlight.svg?branch=master)](https://travis-ci.org/beltegeuse/rustlight)

Physically-based rendering engine implemented with **Rust**.

## Building

NOTE: Need Rust 1.25 at least to support ```repr(align(X))``` routine for embree-rs. To install this version, you can run the following command:

```RUSTUP_DIST_SERVER=https://dev-static.rust-lang.org rustup update stable```

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

## Features

For now, these are the following features implemented:
- Integrators: 
    * ambiant occlusion
    * direct with MIS
    * path-tracing with NEE
    * gradient-path tracing [1]
    * PSSMLT [2]
- Explicit Integrators: Uses a graph to represent the light transport
    * path-tracing with NEE (*2 slower~ than non-explicit one)
    * light-tracing
    * VPL
- Filtering: 
    * image-space control variate with uniform and variance-based weights [3]
- Materials: 
    * diffuse
    * phong lobe
    * specular
- Emitters: 
    * multiple surface lights support

## Rendering

![Cornel Box gradient-domain pt](http://beltegeuse.s3-website-ap-northeast-1.amazonaws.com/rustlight/cbox_gpt_weighted.png)

This image have been rendered using gradient-domain path tracing with image-domain control variate reconstruction (weighted) with 16 samples per pixels (rendering time: ~15 sec on Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz).

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