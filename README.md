# Practical Product Sampling For Single Scattering In Media

Keven Villeneuve&ast;, Adrien Gruson&ast;, Iliyan Georgiev, and Derek Nowrouzezahrai

![ex](https://github.com/beltegeuse/rustlight/actions/workflows/main.yml/badge.svg?branch=point-normal)

Implementation of _"Practical Product Sampling For Single Scattering In Media"_, Villeneuve et al. **EGSR 2021** (Digital Library Only Track). For more information, please check [Keven's website](https://kevenv.github.io/index.html).

## Getting started

Dependencies needed to be installed on your system (tested only on Linux):
- **Rust** (`1.53.0` stable): other versions of Rust should work without issues. To install Rust on your system, see https://rustup.rs/.
- **OpenEXR** (version 3): The code also works with version 2 by modifying `Cargo.toml` file (see comments inside this file directly).
- **Embree** (version 3).
- **Python** (version 3). Only required for `run.py`. 

## Reproducing results

```shell
echo "Download code..."
git clone https://github.com/beltegeuse/rustlight.git
cd rustlight
git checkout point-normal
echo "Building code ..."
cargo build --release --example=cli --features="embree openexr" --
echo "Get scenes and references..." 
wget http://beltegeuse.s3-website-ap-northeast-1.amazonaws.com/research/2021_PointNormal/point_normal_scenes.tar.xz
tar -xvf point_normal_scenes.tar.xz 
echo "Run results Fig. 3, 6, 7..."
mkdir results
python run.py
echo "Run results Fig. 5..."
sh run_plane_exp.sh
```

## Code organization

- `src/integrators/explicit/point_normal.rs`: different implementation of distance sampling (transmittance, equiangular, ours)
- `src/integrators/explicit/point_normal_poly.rs`: Taylor expansion of transmittance/phase order 4 and 6.
- `examples/cli.rs`: command-line interface (search `point_normal`).
- `src/emitter.rs`: ATS implementation (ported from [PBRT-v4](https://github.com/mmp/pbrt-v4))

## Cite

```
@article{Villeneuve:2021:VolumeProductSampling,
  author = {Keven Villeneuve and Adrien Gruson and Iliyan Georgiev and Derek Nowrouzezahrai},
  title = {Practical product sampling for single scattering in media},
  booktitle = {Proceedings of EGSR - Digital Library Only Track},
  year = {2021},
  month = jun,
  publisher = {The Eurographics Association}
}
```
## Other information

Please look at README.md in the master branch.
