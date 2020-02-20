<h1>
Rustlight <img src="http://beltegeuse.s3-website-ap-northeast-1.amazonaws.com/rustlight/logo.png" width="96"> 
</h1>

[![Build Status](https://travis-ci.org/beltegeuse/rustlight.svg?branch=master)](https://travis-ci.org/beltegeuse/rustlight)

Physically-based rendering engine implemented with **Rust**.

## How to use it

```
$ cargo run --release -- -h
rustlight 0.2.0
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
    -m <medium>             add medium with defined density [default: 0.0]
    -n <nbsamples>          integration technique
    -t <nbthreads>          number of thread for the computation [default: auto]
    -o <output>             output image file

ARGS:
    <scene>    JSON file description

SUBCOMMANDS:
    ao                           ambiant occlusion
    direct                       direct lighting
    gradient-path                gradient path tracing
    gradient-path-explicit       gradient path tracing
    help                         Prints this message or the help of the given subcommand(s)
    light                        light tracing generating path from the lights
    path                         path tracing generating path from the sensor
    path_kulla                   path tracing for single scattering
    plane_single                 Prototype implementation of 'Photon surfaces for robust, unbiased volumetric
                                 density estimation'
    pssmlt                       path tracing with MCMC sampling
    uncorrelated_plane_single    Prototype implementation of 'Photon surfaces for robust, unbiased volumetric
                                 density estimation'
    vol_primitives               BRE/Beam/Planes estimators
    vpl                          brute force virtual point light integrator
```

For example, to use path tracing using 128 spp:
```
$ cargo run --release --features="pbrt openexr" -- -a inf -n 128 -o path.pfm ./data/cbox.json path
```

## Dependencies

Optionals : 

- [image](https://github.com/image-rs/image) : load and save LDR images
- [openexr](https://github.com/cessen/openexr-rs) : load and save EXR images
- [embree-rs](https://github.com/Twinklebear/embree-rs) : fast primitive/ray intersection (* not yet optional)
- [pbrt_rs](https://github.com/beltegeuse/pbrt_rs) : read PBRT files 

## Features

For now, these are the following features implemented:
- Integrators (most of them using a common graph to represent the light transport): 
    * Ambiant occlusion
    * Direct with MIS
    * Path-tracing with NEE
    * [*] Gradient-path tracing [1]
    * Primary-sample space MLT [2]
    * Light tracing
    * Virtual Point Light
- Special volumetric integrators (via vol_primitives):
    * Beam radiance estimate (2D kernel) [3]
    * Photon beams (1D kernel) [4]
    * [*] Photon planes (0D kernel) [5]
    * [*] Naive Virtual ray light [6]
- Special single scattering intergrators:
    * (Un)correlated photon planes [7]
    * Kulla importance sampling [8]
- Filtering: 
    * Image-space control variate with uniform and variance-based weights [7]
- Materials: 
    * Diffuse
    * Phong lobe
    * Specular
    * A subset of PBRT materials (imported from [rs_pbrt](https://github.com/wahn/rs_pbrt))
- Emitters: 
    * Multiple tri-mesh lights support
- Volumes:
    * Infinite homogenous participating media
- Phase functions:
    * Isotropic

Techniques with [*] might contains bug or are incomplete (only naive implementation)

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
[1] Kettunen et al. "Gradient-domain path tracing" (SIGGRAPH 2015) \
[2] Csaba et al. "A simple and robust mutation strategy for the metropolis light transport algorithm. (CGF 2002) \
[3] Jarosz et al. "The beam radiance estimate for volumetric photon mapping" (EG 2008) \
[4] Jarosz et al. "Progressive photon beams" (SIGGRAPH Asia 2011) \
[5] Bitterli and Jarosz "Beyond points and beams: Higher-dimensional photon samples for volumetric light transport" (SIGGRAPH 2017) \
[6] Novak et al. "Virtual ray lights for rendering scenes with participating media" (SIGGRAPH 2012) \
[7] Rousselle et al. "Image-space control variates for rendering" (SIGGRAPH 2016) \
[8] Deng et al. "Photon surfaces for robust, unbiased volumetric density estimation" (SIGGRAPH 2019) \
[9] Kulla et al. "Importance Sampling Techniques for Path Tracing in Participating Media" (EGSR 2012)
