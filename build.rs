use std::env;
use std::path::PathBuf;


fn main() {
    // Try to see if env entry
    // have been set for embree
    match env::var("EMBREE_DIR")
    {
        Ok(n) => {
            let mut embree_dir = PathBuf::from(n);
            embree_dir.push("lib");
            println!("cargo:rustc-link-search=native={}", embree_dir.display());
        },
        Err(e) => (),
    }
    // add embree for the linker
    println!("cargo:rustc-link-lib=embree");
}

