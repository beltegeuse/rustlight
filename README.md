# rustlight

[![Build Status](https://travis-ci.org/beltegeuse/rustlight.svg?branch=master)](https://travis-ci.org/beltegeuse/rustlight)

Physically-based rendering engine implemented with **Rust**.

## Features

For now, only path tracing with next event estimation is implemented. These are the following features implemented:

- Materials: Diffuse and Phong lobe
- Emitters: multiple surface lights support

## Rendering 

![Cornel Box](http://beltegeuse.s3-website-ap-northeast-1.amazonaws.com/rustlight/cbox.png)

## Roadmap

Rendering algorithms for path-tracing:
- Gradient-domain path tracing with image-space control variate reconstruction.
- MCMC using PSSMLT

Other rendering features: 

- Materials: Specular (mirror and glass), microfacet with Beckert distribution.
- Emitters: Environmental lights
- Tools: Blender exporter script (based on cycle)

## Inspirations

This code has been inspired from several repositories:

- rs_pbrt project: https://github.com/wahn/rs_pbrt
- the blog post from Brook Heisler: https://bheisler.github.io/post/writing-raytracer-in-rust-part-1/
- tray_rust project: https://github.com/Twinklebear/tray_rust
