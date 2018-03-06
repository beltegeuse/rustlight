# rustlight

[![Build Status](https://travis-ci.org/beltegeuse/rustlight.svg?branch=master)](https://travis-ci.org/wahn/rs_pbrt)

Physically-based rendering engine implemented with **Rust**.

## Cornel box

For now, only path tracing with next event estimation is implemented. Moreover, only diffuse material and only one surfacique light source is supported.

![Cornel Box](http://beltegeuse.s3-website-ap-northeast-1.amazonaws.com/rustlight/cbox.png)

## Roadmap

Core features: 

- Add more materials (Specular, Dielectric, ...)
- Support multiple light and add environmental light support.
- Add blender script exporter

Rendering algorithms:

- PSSMLT (with only path tracing)
- BDPT
- (Stocastic) Progressive photon mapping

## Inspirations

The code have been inspired from several repositories:

- rs_pbrt project: https://github.com/wahn/rs_pbrt
- blog post from Brook Heisler: https://bheisler.github.io/post/writing-raytracer-in-rust-part-1/
- tray_rust project: https://github.com/Twinklebear/tray_rust