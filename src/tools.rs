use byteorder::{LittleEndian, WriteBytesExt};
use cgmath::Point2;
use image::{DynamicImage, GenericImage, PNG};
use integrators::Bitmap;
use std;
use std::fs::File;
use std::io::Write;
use std::iter::Iterator;
use std::path::Path;
use openexr;

pub fn save(imgout_path_str: &str, img: &Bitmap, name: &str) {
    let output_ext = match std::path::Path::new(imgout_path_str).extension() {
        None => panic!("No file extension provided"),
        Some(x) => std::ffi::OsStr::to_str(x).expect("Issue to unpack the file"),
    };
    match output_ext {
        "pfm" => {
            save_pfm(imgout_path_str, img, &name);
        }
        "png" => {
            save_png(imgout_path_str, img, &name);
        },
        "exr" => {
            save_exr(imgout_path_str, img, &name);
        }
        _ => panic!("Unknow output file extension"),
    }
}

pub fn save_exr(imgout_path_str: &str, img: &Bitmap, name: &str) {
    // Pixel data for floating point RGB image.
    let mut pixel_data = vec![];
    pixel_data.reserve((img.size.x * img.size.y) as usize);
    for y in 0..img.size.y {
        for x in 0..img.size.x {
            let p = Point2::new(x, y);
            let c = img.get(p, name);
            pixel_data.push((c.r,c.g,c.b));
        }
    }

    // Create a file to write to.  The `Header` determines the properties of the
    // file, like resolution and what channels it has.
    let mut file = std::fs::File::create(Path::new(imgout_path_str)).unwrap();
    let mut output_file = openexr::ScanlineOutputFile::new(
        &mut file,
        openexr::Header::new()
            .set_resolution(img.size.x, img.size.y)
            .add_channel("R", openexr::PixelType::FLOAT)
            .add_channel("G", openexr::PixelType::FLOAT)
            .add_channel("B", openexr::PixelType::FLOAT),
    ).unwrap();

    // Create a `FrameBuffer` that points at our pixel data and describes it as
    // RGB data.
    let mut fb = openexr::FrameBuffer::new(img.size.x, img.size.y);
    fb.insert_channels(&["R", "G", "B"], &pixel_data);

    // Write pixel data to the file.
    output_file.write_pixels(&fb).unwrap();
}

pub fn save_pfm(imgout_path_str: &str, img: &Bitmap, name: &str) {
    let mut file = File::create(Path::new(imgout_path_str)).unwrap();
    let header = format!("PF\n{} {}\n-1.0\n", img.size.y, img.size.x);
    file.write(header.as_bytes()).unwrap();
    for y in 0..img.size.y {
        for x in 0..img.size.x {
            let p = img.get(Point2::new(x, img.size.y - y - 1), name);
            file.write_f32::<LittleEndian>(p.r.abs()).unwrap();
            file.write_f32::<LittleEndian>(p.g.abs()).unwrap();
            file.write_f32::<LittleEndian>(p.b.abs()).unwrap();
        }
    }
}

pub fn save_png(imgout_path_str: &str, img: &Bitmap, name: &str) {
    // The image that we will render
    let mut image_ldr = DynamicImage::new_rgb8(img.size.x, img.size.y);
    for x in 0..img.size.x {
        for y in 0..img.size.y {
            let p = Point2::new(x, y);
            image_ldr.put_pixel(x, y, img.get(p, name).to_rgba())
        }
    }
    let ref mut fout = File::create(imgout_path_str).unwrap();
    image_ldr
        .save(fout, PNG)
        .expect("failed to write img into file");
}

/// For be able to have range iterators
pub struct StepRangeInt {
    start: usize,
    end: usize,
    step: usize,
}

impl StepRangeInt {
    pub fn new(start: usize, end: usize, step: usize) -> StepRangeInt {
        StepRangeInt { start, end, step }
    }
}

impl Iterator for StepRangeInt {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.start < self.end {
            let v = self.start;
            self.start = v + self.step;
            Some(v)
        } else {
            None
        }
    }
}

pub trait ModuloSignedExt {
    fn modulo(&self, n: Self) -> Self;
}
macro_rules! modulo_signed_ext_impl {
    ($($t:ty)*) => ($(
        impl ModuloSignedExt for $t {
            #[inline]
            fn modulo(&self, n: Self) -> Self {
                (self % n + n) % n
            }
        }
    )*)
}
modulo_signed_ext_impl! { f32 }
