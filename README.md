# rustlight

[![Build Status](https://travis-ci.org/beltegeuse/rustlight.svg?branch=master)](https://travis-ci.org/beltegeuse/rustlight)

Physically-based rendering engine implemented with **Rust**.

## Features

For now, these are the following features implemented:
- Integrators: ambiant occlusion, direct, path-tracing and gradient-domain path tracing.
- Filtering: image-domain control variate reconstruction (uniform weights)
- Materials: diffuse and phong lobe
- Emitters: multiple surface lights support

## Rendering 

![Cornel Box gradient-domain pt](http://beltegeuse.s3-website-ap-northeast-1.amazonaws.com/rustlight/cbox_gpt_uni.png)

This image have been rendered using gradient-domain path tracing with image-domain control variate reconstruction (uniform weight) with 16 samples per pixels (rendering time: ~15 sec).

## Roadmap

Rendering algorithms for path-tracing:
- Fixing gradient-domain path tracing: seems to have wrong gradient when the light source is not visible from the base path
- Add weighted control variate reconstruction
- gradient-domain path reuse
- PSSMLT

Other rendering features: 

- Materials: Specular (mirror and glass), microfacet with Beckert distribution.
- Emitters: Environmental lights
- Tools: Blender exporter script (based on cycle)

## Inspirations

This code has been inspired from several repositories:

- rs_pbrt project: https://github.com/wahn/rs_pbrt
- the blog post from Brook Heisler: https://bheisler.github.io/post/writing-raytracer-in-rust-part-1/
- tray_rust project: https://github.com/Twinklebear/tray_rust
