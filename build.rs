use std::env;
use std::path::PathBuf;

fn main() {
    //let mut embree_dir = PathBuf::from(env::var("EMBREE_DIR").unwrap());
    //embree_dir.push("lib");
    //println!("cargo:rustc-link-search=native={}", embree_dir.display());
    println!("cargo:rustc-link-lib=embree");
}

