# rustlight

[![Build Status](https://travis-ci.org/beltegeuse/rustlight.svg?branch=master)](https://travis-ci.org/beltegeuse/rustlight)

Physically-based rendering engine implemented with **Rust**.

## Building

NOTE: Need Rust 1.25 at least to support ```repr(align(X))``` routine for embree-rs. To install this version, you can run the following command:

```RUSTUP_DIST_SERVER=https://dev-static.rust-lang.org rustup update stable```

## How to use it

```
$ cargo run --release -- -h
rustlight 0.0.4
Adrien Gruson <adrien.gruson@gmail.com>
A Rusty Light Transport simulation program

USAGE:
    rustlight [OPTIONS] <scene> [SUBCOMMAND]

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -n <nbsamples>        integration technique
    -t <nbthreads>        number of thread for the computation [default: auto]
    -o <output>           output image file

ARGS:
    <scene>    JSON file description

SUBCOMMANDS:
    ao               ambiant occlusion
    direct           direct lighting
    gd-path          gradient-domain path tracing
    help             Prints this message or the help of the given subcommand(s)
    path             path tracing
    path-explicit    path tracing with explict light path construction
    pssmlt           path tracing with MCMC sampling

```

For example, to use path tracing using 128 spp:
```
$ cargo run --release -- -n 128 -o path.pfm ./data/cbox.json path
```

## Features

For now, these are the following features implemented:
- Integrators: ambiant occlusion, direct, path-tracing, gradient-domain path tracing, PSSMLT
- Explicit path building (generate a sensor path and evaluate it later)
- Filtering: image-domain control variate reconstruction (uniform weights)
- Materials: diffuse, phong lobe, specular
- Emitters: multiple surface lights support

## Rendering

![Cornel Box gradient-domain pt](http://beltegeuse.s3-website-ap-northeast-1.amazonaws.com/rustlight/cbox_gpt_uni.png)

This image have been rendered using gradient-domain path tracing with image-domain control variate reconstruction (uniform weight) with 16 samples per pixels (rendering time: ~15 sec).

## Roadmap

Rendering algorithms for path-tracing:
- Fixing gradient-domain path tracing: seems to have wrong gradient when the light source is not visible from the base path
- Add weighted control variate reconstruction
- gradient-domain path reuse

Other rendering features:

- Materials: glass, microfacet with Beckert distribution.
- Emitters: Environmental and point lights
- Tools: Blender exporter script (based on cycle)

## Inspirations

This code has been inspired from several repositories:

- rs_pbrt project: https://github.com/wahn/rs_pbrt
- the blog post from Brook Heisler: https://bheisler.github.io/post/writing-raytracer-in-rust-part-1/
- tray_rust project: https://github.com/Twinklebear/tray_rust
- mitsuba: https://github.com/mitsuba-renderer/mitsuba
