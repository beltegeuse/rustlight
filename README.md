<h1>
Rustlight <img src="http://beltegeuse.s3-website-ap-northeast-1.amazonaws.com/rustlight/logo.png" width="96"> 
</h1>

[![Build Status](https://travis-ci.org/beltegeuse/rustlight.svg?branch=smis-planes)](https://travis-ci.org/beltegeuse/rustlight)

Physically-based rendering engine implemented with **Rust**.

## How to replicate CMIS Photon planes

These are the instructions to reproduce "Continuous Multiple Importance Sampling", West et al., SIGGRAPH 2020. This repository only contains the code for "Volume single scattering" (Section 6). Below are the instructions:

1) You need to checkout the repository, use the correct branch and download scene information:
```shell
$ git clone https://github.com/beltegeuse/rustlight.git
$ cd rustlight
$ git checkout smis-planes
$ wget http://beltegeuse.s3-website-ap-northeast-1.amazonaws.com/research/2020_CMIS/plane_scene.zip
$ unzip plane_scene.zip
```

2) You can regenerate results of figure 11. Note that the first invocation of cargo will build the project:
```shell
$ export SCENE=scene/meeting_ply.pbrt
$ export NBPLANES=40960
$ cargo run --features="pbrt openexr" --release -- -t -2 -n 1 -o ualpha.exr -m 0.2 $SCENE plane_single -n $NBPLANES -s ualpha
$ cargo run --features="pbrt openexr" --release -- -t -2 -n 1 -o smis_jacobian_k2_stratified.exr -m 0.2 $SCENE plane_single -n $NBPLANES -s smis_jacobian -k 2 -x
$ cargo run --features="pbrt openexr" --release -- -t -2 -n 1 -o smis_all_k2_stratified.exr -m 0.2 $SCENE plane_single -n $NBPLANES -s smis_all -k 2 -x
$ cargo run --features="pbrt openexr" --release -- -t -2 -n 1 -o cmis.exr -m 0.2 $SCENE plane_single -n $NBPLANES -s cmis
```
The precomputed reference is available at: http://beltegeuse.s3-website-ap-northeast-1.amazonaws.com/research/2020_CMIS/plane_reference.exr

For more information about the available options for this particular integrator:
```shell
$ cargo run --release --features="pbrt openexr" -- -t -2 -n 1 -o ualpha.exr -m 0.2 $SCENE plane_single -h
```

## Dependencies

Optionals : 

- [image](https://github.com/image-rs/image) : load and save LDR images
- [openexr](https://github.com/cessen/openexr-rs) : load and save EXR images
- [embree-rs](https://github.com/Twinklebear/embree-rs) : fast primitive/ray intersection (* not yet optional)
- [pbrt_rs](https://github.com/beltegeuse/pbrt_rs) : read PBRT files 

## More information

Please look at README.md in the master branch.
