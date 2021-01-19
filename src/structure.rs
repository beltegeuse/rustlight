use crate::constants;
use crate::geometry::Mesh;
use crate::math::Frame;
use crate::tools::*;
use crate::Scale;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use cgmath::{EuclideanSpace, InnerSpace, Point2, Point3, Vector2, Vector3};
#[cfg(feature = "image")]
use image::{DynamicImage, GenericImage, Pixel};
#[cfg(feature = "openexr")]
use openexr;
use std;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::ops::*;
use std::path::Path;

#[derive(Clone, Debug)]
pub enum PDF {
    SolidAngle(f32),
    Area(f32),
    Discrete(f32),
}
impl PDF {
    // g_ad = cos(x) / d^2
    pub fn as_solid_angle_geom(self, g_ad: f32) -> Self {
        match self {
            PDF::SolidAngle(v) => PDF::SolidAngle(v),
            PDF::Area(v) => {
                if g_ad == 0.0 {
                    PDF::SolidAngle(0.0)
                } else {
                    PDF::SolidAngle(v / g_ad)
                }
            }
            PDF::Discrete(_) => panic!("Try to convert discrete to SA"),
        }
    }
    pub fn as_solid_angle(self, p: &Point3<f32>, x: &Point3<f32>, n_x: &Vector3<f32>) -> Self {
        match self {
            PDF::SolidAngle(v) => PDF::SolidAngle(v),
            PDF::Area(v) => {
                let d = x - p;
                let l = d.magnitude();
                assert_ne!(l, 0.0);
                let d = d / l;
                let cos = n_x.dot(d).max(0.0);
                if cos == 0.0 {
                    PDF::SolidAngle(0.0)
                } else {
                    PDF::SolidAngle(v * l * l / cos)
                }
            }
            PDF::Discrete(_) => panic!("Try to convert discrete to SA"),
        }
    }
}

#[derive(PartialEq, Clone, Copy)]
pub enum Domain {
    SolidAngle,
    Discrete,
}
#[derive(PartialEq, Clone, Copy)]
pub enum Transport {
    Importance, //< From the camera
    Radiance,   //< From the light
}

impl PDF {
    pub fn is_zero(&self) -> bool {
        match self {
            PDF::Discrete(v) | PDF::SolidAngle(v) | PDF::Area(v) => (*v == 0.0),
        }
    }

    pub fn value(&self) -> f32 {
        match self {
            PDF::Discrete(v) | PDF::SolidAngle(v) | PDF::Area(v) => *v,
        }
    }
}

impl Mul<f32> for PDF {
    type Output = PDF;
    fn mul(self, other: f32) -> PDF {
        match self {
            PDF::Area(v) => PDF::Area(v * other),
            PDF::Discrete(v) => PDF::Discrete(v * other),
            PDF::SolidAngle(v) => PDF::SolidAngle(v * other),
        }
    }
}

pub struct SampledPosition {
    pub p: Point3<f32>,
    pub n: Vector3<f32>,
    pub pdf: PDF,
}

/// Pixel color representation
#[derive(Clone, PartialEq, Debug, Copy)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Color {
    pub fn new(r: f32, g: f32, b: f32) -> Color {
        Color { r, g, b }
    }
    pub fn zero() -> Color {
        Color::new(0.0, 0.0, 0.0)
    }
    pub fn one() -> Color {
        Color::new(1.0, 1.0, 1.0)
    }
    pub fn value(v: f32) -> Color {
        Color::new(v, v, v)
    }
    pub fn abs(&self) -> Color {
        Color::new(self.r.abs(), self.g.abs(), self.b.abs())
    }
    pub fn sqrt(self) -> Color {
        Color::new(self.r.sqrt(), self.g.sqrt(), self.b.sqrt())
    }
    pub fn safe_sqrt(self) -> Color {
        Color::new(
            self.r.max(0.0).sqrt(),
            self.g.max(0.0).sqrt(),
            self.b.max(0.0).sqrt(),
        )
    }
    pub fn avg(&self) -> f32 {
        (self.r + self.g + self.b) / 3.0
    }
    pub fn exp(self) -> Color {
        Color::new(self.r.exp(), self.g.exp(), self.b.exp())
    }
    pub fn get(&self, c: u8) -> f32 {
        match c {
            0 => self.r,
            1 => self.g,
            2 => self.b,
            _ => unimplemented!("Impossible to have more than 3 channels"),
        }
    }

    pub fn is_zero(&self) -> bool {
        self.r == 0.0 && self.g == 0.0 && self.b == 0.0
    }
    pub fn is_valid(&self) -> bool {
        self.r >= 0.0 && self.g >= 0.0 && self.b >= 0.0
    }

    #[cfg(feature = "image")]
    pub fn to_rgba(&self) -> image::Rgba<u8> {
        image::Rgba::from_channels(
            (self.r.min(1.0).powf(1.0 / 2.2) * 255.0) as u8,
            (self.g.min(1.0).powf(1.0 / 2.2) * 255.0) as u8,
            (self.b.min(1.0).powf(1.0 / 2.2) * 255.0) as u8,
            255,
        )
    }
    pub fn channel_max(&self) -> f32 {
        self.r.max(self.g.max(self.b))
    }

    pub fn luminance(&self) -> f32 {
        // FIXME: sRGB??
        self.r * 0.212_671 + self.g * 0.715_160 + self.b * 0.072_169
    }
}

impl Default for Color {
    fn default() -> Self {
        Color::zero()
    }
}

impl Scale<f32> for Color {
    fn scale(&mut self, v: f32) {
        self.r *= v;
        self.g *= v;
        self.b *= v;
    }
}

impl Neg for Color {
    type Output = Color;
    fn neg(self) -> Self::Output {
        Color::new(-self.r, -self.g, -self.b)
    }
}

/////////////// Operators
impl DivAssign<f32> for Color {
    fn div_assign(&mut self, other: f32) {
        self.r /= other;
        self.g /= other;
        self.b /= other;
    }
}

impl<'b> MulAssign<Color> for Color {
    fn mul_assign(&mut self, other: Color) {
        self.r *= other.r;
        self.g *= other.g;
        self.b *= other.b;
    }
}

impl<'b> MulAssign<&'b Color> for Color {
    fn mul_assign(&mut self, other: &'b Color) {
        self.r *= other.r;
        self.g *= other.g;
        self.b *= other.b;
    }
}

impl MulAssign<f32> for Color {
    fn mul_assign(&mut self, other: f32) {
        self.r *= other;
        self.g *= other;
        self.b *= other;
    }
}

impl<'b> AddAssign<&'b Color> for Color {
    fn add_assign(&mut self, other: &'b Color) {
        self.r += other.r;
        self.g += other.g;
        self.b += other.b;
    }
}

impl AddAssign<Color> for Color {
    fn add_assign(&mut self, other: Color) {
        self.r += other.r;
        self.g += other.g;
        self.b += other.b;
    }
}

impl Div<f32> for Color {
    type Output = Self;
    fn div(self, other: f32) -> Color {
        //assert!();
        if other == 0.0 || !other.is_finite() {
            warn!("0 f32 division detected!");
            Color::zero()
        } else {
            Color {
                r: self.r / other,
                g: self.g / other,
                b: self.b / other,
            }
        }
        //assert_ne!(other, 0.0);
    }
}

impl Div<Color> for Color {
    type Output = Self;
    fn div(self, other: Color) -> Color {
        Color {
            r: self.r / other.r,
            g: self.g / other.g,
            b: self.b / other.b,
        }
    }
}

impl Mul<f32> for Color {
    type Output = Self;
    fn mul(self, other: f32) -> Color {
        //assert!(other.is_finite());
        if other.is_finite() {
            Color {
                r: self.r * other,
                g: self.g * other,
                b: self.b * other,
            }
        } else {
            Color::zero()
        }
    }
}

impl Mul<Color> for f32 {
    type Output = Color;
    fn mul(self, other: Color) -> Color {
        Color {
            r: other.r * self,
            g: other.g * self,
            b: other.b * self,
        }
    }
}

impl<'a, 'b> Sub<&'a Color> for &'b Color {
    type Output = Color;
    fn sub(self, other: &'a Color) -> Color {
        Color {
            r: other.r - self.r,
            g: other.g - self.g,
            b: other.b - self.b,
        }
    }
}

impl<'a> Mul<&'a Color> for f32 {
    type Output = Color;
    fn mul(self, other: &'a Color) -> Color {
        Color {
            r: other.r * self,
            g: other.g * self,
            b: other.b * self,
        }
    }
}

impl<'a> Mul<&'a Color> for Color {
    type Output = Self;
    fn mul(self, other: &'a Color) -> Color {
        Color {
            r: self.r * other.r,
            g: self.g * other.g,
            b: self.b * other.b,
        }
    }
}

impl Mul<Color> for Color {
    type Output = Self;
    fn mul(self, other: Color) -> Color {
        Color {
            r: self.r * other.r,
            g: self.g * other.g,
            b: self.b * other.b,
        }
    }
}

impl Sub<Color> for Color {
    type Output = Self;
    fn sub(self, other: Color) -> Color {
        Color {
            r: self.r - other.r,
            g: self.g - other.g,
            b: self.b - other.b,
        }
    }
}

impl Add<Color> for Color {
    type Output = Self;
    fn add(self, other: Color) -> Color {
        Color {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
        }
    }
}

impl<'a> Add<&'a Color> for Color {
    type Output = Self;
    fn add(self, other: &'a Color) -> Color {
        Color {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
        }
    }
}

#[derive(Clone)]
pub struct Bitmap {
    pub size: Vector2<u32>,
    pub colors: Vec<Color>,
}
impl Bitmap {
    pub fn new(size: Vector2<u32>) -> Bitmap {
        Bitmap {
            size,
            colors: vec![Color::default(); (size.x * size.y) as usize],
        }
    }
    pub fn clear(&mut self) {
        self.colors.iter_mut().for_each(|x| *x = Color::default());
    }
    pub fn accumulate(&mut self, p: Point2<u32>, f: Color) {
        assert!(p.x < self.size.x);
        assert!(p.y < self.size.y);
        let index = (p.y * self.size.x + p.x) as usize;
        self.colors[index] += f;
    }
    /**
     * pos: Position where to splat the buffer
     */
    pub fn accumulate_bitmap(&mut self, o: &Bitmap, pos: Point2<u32>) {
        for y in 0..o.size.y {
            for x in 0..o.size.x {
                let p = Point2::new(pos.x + x, pos.y + y);
                let index = (p.y * self.size.x + p.x) as usize;
                let index_other = (y * o.size.x + x) as usize;
                self.colors[index] += o.colors[index_other];
            }
        }
    }
    pub fn gamma(&mut self, v: f32) {
        for c in &mut self.colors {
            c.r = c.r.powf(v);
            c.g = c.g.powf(v);
            c.b = c.b.powf(v);
        }
    }
    pub fn scale(&mut self, v: f32) {
        self.colors.iter_mut().for_each(|x| x.scale(v));
    }
    pub fn average(&self) -> Color {
        let mut s = Color::default();
        self.colors.iter().for_each(|x| s += x);
        s.scale(1.0 / self.colors.len() as f32);
        s
    }

    // Get the pixel value at the given position
    pub fn pixel_uv(&self, mut uv: Vector2<f32>) -> Color {
        uv.x = uv.x.modulo(1.0);
        uv.y = uv.y.modulo(1.0);
        let (x, y) = (
            (uv.x * self.size.x as f32) as usize,
            (uv.y * self.size.y as f32) as usize,
        );
        let i = self.size.x as usize * y + x;
        if i >= self.colors.len() {
            warn!(
                "Exceed UV coordinates: {:?} | {:?} | {:?}",
                uv,
                self.size,
                (x, y)
            );
            Color::default()
        } else {
            self.colors[i]
        }
    }

    pub fn pixel(&self, p: Point2<u32>) -> Color {
        assert!(p.x < self.size.x);
        assert!(p.y < self.size.y);
        self.colors[(p.y * self.size.x + p.x) as usize]
    }
    pub fn pixel_mut(&mut self, p: Point2<u32>) -> &mut Color {
        assert!(p.x < self.size.x);
        assert!(p.y < self.size.y);
        &mut self.colors[(p.y * self.size.x + p.x) as usize]
    }

    // Save functions
    #[cfg(not(feature = "image"))]
    pub fn save_ldr_image(&self, _imgout_path_str: &str) {
        panic!("Rustlight wasn't built with Image support.");
    }
    #[cfg(feature = "image")]
    pub fn save_ldr_image(&self, imgout_path_str: &str) {
        // The image that we will render
        let mut image_ldr = DynamicImage::new_rgb8(self.size.x, self.size.y);
        for x in 0..self.size.x {
            for y in 0..self.size.y {
                let p = Point2::new(x, y);
                image_ldr.put_pixel(x, y, self.pixel(p).to_rgba())
            }
        }
        image_ldr
            .save(&Path::new(imgout_path_str))
            .expect("failed to write img into file");
    }

    #[cfg(not(feature = "openexr"))]
    pub fn save_exr(&self, _imgout_path_str: &str) {
        panic!("Rustlight wasn't built with OpenExr support.");
    }
    #[cfg(feature = "openexr")]
    pub fn save_exr(&self, imgout_path_str: &str) {
        // Pixel data for floating point RGB image.
        let mut pixel_data = vec![];
        pixel_data.reserve((self.size.x * self.size.y) as usize);
        for y in 0..self.size.y {
            for x in 0..self.size.x {
                let p = Point2::new(x, y);
                let c = self.pixel(p);
                pixel_data.push((c.r, c.g, c.b));
            }
        }

        // Create a file to write to.  The `Header` determines the properties of the
        // file, like resolution and what channels it has.
        let mut file = std::fs::File::create(Path::new(imgout_path_str)).unwrap();
        let mut output_file = openexr::ScanlineOutputFile::new(
            &mut file,
            openexr::Header::new()
                .set_resolution(self.size.x, self.size.y)
                .add_channel("R", openexr::PixelType::FLOAT)
                .add_channel("G", openexr::PixelType::FLOAT)
                .add_channel("B", openexr::PixelType::FLOAT),
        )
        .unwrap();

        // Create a `FrameBuffer` that points at our pixel data and describes it as
        // RGB data.
        let fb = {
            // Create the frame buffer
            let mut fb = openexr::FrameBuffer::new(self.size.x, self.size.y);
            fb.insert_channels(&["R", "G", "B"], &pixel_data);
            fb
        };

        // Write pixel data to the file.
        output_file.write_pixels(&fb).unwrap();
    }
    pub fn save(&self, imgout_path_str: &str) {
        let output_ext = match std::path::Path::new(imgout_path_str).extension() {
            None => panic!("No file extension provided"),
            Some(x) => std::ffi::OsStr::to_str(x).expect("Issue to unpack the file"),
        };
        match output_ext {
            "pfm" => {
                self.save_pfm(imgout_path_str);
            }
            "png" => {
                self.save_ldr_image(imgout_path_str);
            }
            "exr" => {
                self.save_exr(imgout_path_str);
            }
            _ => panic!("Unknow output file extension"),
        }
    }

    pub fn save_pfm(&self, imgout_path_str: &str) {
        let file = File::create(Path::new(imgout_path_str)).unwrap();
        let mut file = BufWriter::new(file);
        let header = format!("PF\n{} {}\n-1.0\n", self.size.x, self.size.y);
        file.write_all(header.as_bytes()).unwrap();
        for y in 0..self.size.y {
            for x in 0..self.size.x {
                let p = self.pixel(Point2::new(x, self.size.y - y - 1));
                file.write_f32::<LittleEndian>(p.r.abs()).unwrap();
                file.write_f32::<LittleEndian>(p.g.abs()).unwrap();
                file.write_f32::<LittleEndian>(p.b.abs()).unwrap();
            }
        }
    }

    // Load images
    pub fn read_pfm(filename: &str) -> Self {
        dbg!(filename);
        let f = File::open(Path::new(filename)).unwrap();
        let mut f = BufReader::new(f);
        // Check the flag
        {
            let mut header_str = String::new();
            f.read_line(&mut header_str).unwrap();
            if header_str != "PF\n" {
                panic!("Wrong PF flag encounter");
            }
        }
        // Check the dim
        let size = {
            let mut img_dim = String::new();
            f.read_line(&mut img_dim).unwrap();
            let img_dim = img_dim
                .split(" ")
                .map(|v| v.trim().parse::<u32>().unwrap())
                .collect::<Vec<_>>();
            assert!(img_dim.len() == 2);
            Vector2::new(img_dim[0], img_dim[1])
        };

        // Check the encoding
        {
            let mut encoding = String::new();
            f.read_line(&mut encoding).unwrap();
            let encoding = encoding.trim().parse::<f32>().unwrap();
            assert!(encoding == -1.0);
        }

        let mut colors = vec![Color::zero(); (size.x * size.y) as usize];
        for y in 0..size.y {
            for x in 0..size.x {
                let p = Point2::new(x, size.y - y - 1);
                let r = f.read_f32::<LittleEndian>().unwrap();
                let g = f.read_f32::<LittleEndian>().unwrap();
                let b = f.read_f32::<LittleEndian>().unwrap();
                colors[(p.y * size.x + p.x) as usize] = Color::new(r, g, b);
            }
        }

        Bitmap { size, colors }
    }
    #[cfg(not(feature = "openexr"))]
    pub fn read_exr(_filename: &str) -> Self {
        panic!("Rustlight wasn't built with OpenEXR support");
    }
    #[cfg(feature = "openexr")]
    pub fn read_exr(filename: &str) -> Self {
        // Open the EXR file.
        let mut file = std::fs::File::open(filename).unwrap();
        let mut input_file = openexr::InputFile::new(&mut file).unwrap();

        // Get the image dimensions, so we know how large of a buffer to make.
        let (width, height) = input_file.header().data_dimensions();
        let size = Vector2::new(width, height);

        // Buffer to read pixel data into.
        let mut pixel_data = vec![(0.0f32, 0.0f32, 0.0f32); (width * height) as usize];

        // New scope because `FrameBuffer` mutably borrows `pixel_data`, so we need
        // it to go out of scope before we can access our `pixel_data` again.
        {
            // Create `FrameBufferMut` that points at our pixel data and describes
            // it as RGB data.
            let mut fb = openexr::FrameBufferMut::new(width, height);
            fb.insert_channels(&[("R", 0.0), ("G", 0.0), ("B", 0.0)], &mut pixel_data);

            // Read pixel data from the file.
            input_file.read_pixels(&mut fb).unwrap();
        }

        let colors = pixel_data
            .into_iter()
            .map(|v| Color::new(v.0, v.1, v.2))
            .collect::<Vec<Color>>();
        Bitmap { size, colors }
    }
    #[cfg(not(feature = "image"))]
    pub fn read_ldr_image(_filename: &str) -> Self {
        panic!("Rustlight wasn't built with image support");
        Bitmap::default()
    }
    #[cfg(feature = "image")]
    pub fn read_ldr_image(filename: &str) -> Self {
        // The image that we will render
        let image_ldr = image::open(filename)
            .unwrap_or_else(|_| panic!("Impossible to read image: {}", filename));
        let image_ldr = image_ldr.to_rgb8();
        let size = Vector2::new(image_ldr.width(), image_ldr.height());
        let mut colors = vec![Color::zero(); (size.x * size.y) as usize];
        for x in 0..size.x {
            for y in 0..size.y {
                let p = image_ldr.get_pixel(x, y);
                colors[(y * size.x + x) as usize] = Color::new(
                    f32::from(p[0]) / 255.0,
                    f32::from(p[1]) / 255.0,
                    f32::from(p[2]) / 255.0,
                );
            }
        }

        Bitmap { size, colors }
    }

    pub fn read(filename: &str) -> Self {
        let ext = match std::path::Path::new(filename).extension() {
            None => panic!("No file extension provided"),
            Some(x) => std::ffi::OsStr::to_str(x).expect("Issue to unpack the file"),
        };
        match ext {
            "pfm" => Bitmap::read_pfm(filename),
            "exr" => Bitmap::read_exr(filename),
            _ => {
                // Try the default implementation support
                Bitmap::read_ldr_image(filename)
            }
        }
    }
}
// By default, create a black image
impl Default for Bitmap {
    fn default() -> Self {
        Bitmap {
            size: Vector2::new(1, 1),
            colors: vec![Color::zero()],
        }
    }
}

/// Ray representation
#[derive(Clone)]
pub struct Ray {
    pub o: Point3<f32>,
    pub d: Vector3<f32>,
    pub tnear: f32,
    pub tfar: f32,
}

impl Ray {
    pub fn new(o: Point3<f32>, d: Vector3<f32>) -> Ray {
        // TODO: Check if this assert is not too costly.
        assert_approx_eq!(d.dot(d), 1.0, 0.0001);
        Ray {
            o,
            d,
            tnear: constants::EPSILON,
            tfar: std::f32::MAX,
        }
    }
}

// Some function based on vectors
fn vec_min(v1: &Vector3<f32>, v2: &Vector3<f32>) -> Vector3<f32> {
    Vector3::new(v1.x.min(v2.x), v1.y.min(v2.y), v1.z.min(v2.z))
}

fn vec_max(v1: &Vector3<f32>, v2: &Vector3<f32>) -> Vector3<f32> {
    Vector3::new(v1.x.max(v2.x), v1.y.max(v2.y), v1.z.max(v2.z))
}

fn vec_div(v1: &Vector3<f32>, v2: &Vector3<f32>) -> Vector3<f32> {
    Vector3::new(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z)
}

fn vec_mult(v1: &Vector3<f32>, v2: &Vector3<f32>) -> Vector3<f32> {
    Vector3::new(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z)
}

fn vec_max_coords(v: Vector3<f32>) -> f32 {
    v.x.max(v.y.max(v.z))
}

fn vec_min_coords(v: Vector3<f32>) -> f32 {
    v.x.min(v.y.min(v.z))
}

#[derive(Debug)]
pub struct AABB {
    pub p_min: Vector3<f32>,
    pub p_max: Vector3<f32>,
}

impl Default for AABB {
    fn default() -> Self {
        Self {
            p_min: Vector3::new(std::f32::MAX, std::f32::MAX, std::f32::MAX),
            p_max: Vector3::new(std::f32::MIN, std::f32::MIN, std::f32::MIN),
        }
    }
}

impl AABB {
    pub fn union_aabb(&self, b: &AABB) -> AABB {
        AABB {
            p_min: vec_min(&self.p_min, &b.p_min),
            p_max: vec_max(&self.p_max, &b.p_max),
        }
    }

    pub fn union_vec(&self, v: &Vector3<f32>) -> AABB {
        AABB {
            p_min: vec_min(&self.p_min, v),
            p_max: vec_max(&self.p_max, v),
        }
    }

    pub fn size(&self) -> Vector3<f32> {
        self.p_max - self.p_min
    }

    pub fn center(&self) -> Vector3<f32> {
        self.size() * 0.5 + self.p_min
    }

    pub fn intersect(&self, r: &Ray) -> Option<f32> {
        // TODO: direction inverse could be precomputed
        let t_0 = vec_div(&(self.p_min - r.o.to_vec()), &r.d);
        let t_1 = vec_div(&(self.p_max - r.o.to_vec()), &r.d);
        let t_min = vec_max_coords(vec_min(&t_0, &t_1));
        let t_max = vec_min_coords(vec_max(&t_0, &t_1));
        if t_min <= t_max {
            // FIXME: Maybe wrong if tmin is different
            if t_min >= r.tfar {
                None
            } else {
                Some(t_min)
            }
        } else {
            None
        }
    }

    pub fn to_sphere(&self) -> BoundingSphere {
        let c = self.center();
        BoundingSphere {
            center: Point3::new(c.x, c.y, c.z),
            radius: (c - self.p_max).magnitude(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BoundingSphere {
    pub center: Point3<f32>,
    pub radius: f32,
}
impl Default for BoundingSphere {
    fn default() -> Self {
        Self {
            center: Point3::new(0.0, 0.0, 0.0),
            radius: 0.0,
        }
    }
}

impl BoundingSphere {
    pub fn is_empty(&self) -> bool {
        self.radius <= 0.0
    }

    pub fn intersect(&self, r: &Ray) -> Option<f32> {
        let d_p = self.center - r.o;
        let a = r.d.magnitude2();
        let b = 2.0 * d_p.dot(r.d);
        let c = d_p.magnitude2() - self.radius * self.radius;

        if let Some((t0, t1)) = crate::math::solve_quadratic(a, b, c) {
            if t0 < r.tnear {
                if t1 < r.tfar {
                    Some(t1)
                } else {
                    None
                }
            } else if t0 < r.tfar {
                Some(t0)
            } else {
                None
            }
        } else {
            None
        }
    }
}

// Simple intersection primitive
pub struct IntersectionUV {
    pub t: f32,
    pub p: Point3<f32>,
    pub n: Vector3<f32>,
    pub u: f32,
    pub v: f32,
}

#[derive(Clone)]
pub struct Intersection<'a> {
    /// Intersection distance
    pub dist: f32,
    /// Geometry normal
    pub n_g: Vector3<f32>,
    /// Shading normal
    pub n_s: Vector3<f32>,
    /// Intersection point
    pub p: Point3<f32>,
    /// Textures coordinates
    pub uv: Option<Vector2<f32>>,
    /// Mesh which we have intersected
    pub mesh: &'a Mesh,
    /// Frame from the intersection point
    pub frame: Frame,
    /// Incomming direction in the local coordinates
    pub wi: Vector3<f32>,
}

impl<'a> Intersection<'a> {
    pub fn cos_theta(&self) -> f32 {
        self.wi.z
    }
    pub fn to_local(&self, d: &Vector3<f32>) -> Vector3<f32> {
        self.frame.to_local(*d)
    }
    pub fn to_world(&self, d: &Vector3<f32>) -> Vector3<f32> {
        self.frame.to_world(*d)
    }
    pub fn fill_intersection(
        mesh: &'a crate::geometry::Mesh,
        tri_id: usize,
        hit_u: f32,
        hit_v: f32,
        ray: &Ray,
        n_g: Vector3<f32>,
        dist: f32,
        p: Point3<f32>,
    ) -> Intersection<'a> {
        let index = mesh.indices[tri_id];

        let n_s = if let Some(normals) = &mesh.normals {
            let d0 = &normals[index.x];
            let d1 = &normals[index.y];
            let d2 = &normals[index.z];
            let mut n_s = d0 * (1.0 - hit_u - hit_v) + d1 * hit_u + d2 * hit_v;
            if n_g.dot(n_s) < 0.0 {
                n_s = -n_s;
            }
            // Normalize the shading normal
            // TODO: This can be costly, but we want to be sure that the normal are corrects
            //  But I guess it is the cost to pay to prevent other problem due to bad
            //  geometry. I need to check what are the other strategies that have been used
            //  in other rendering engine.
            let l = n_s.dot(n_s);
            if l != 1.0 {
                n_s / l.sqrt()
            } else {
                n_s
            }
        } else {
            n_g.clone()
        };

        // Hack for two sided surfaces
        // Note that we do not fix the surfaces if:
        //  - the bsdf is not two sided (like glass where the normal orientation gives us extra information)
        //  - if it is a light source
        let (n_s, n_g) =
            if mesh.bsdf.is_twosided() && mesh.emission.is_zero() && ray.d.dot(n_s) > 0.0 {
                (
                    Vector3::new(-n_s.x, -n_s.y, -n_s.z),
                    Vector3::new(-n_g.x, -n_g.y, -n_g.z),
                )
            } else {
                (n_s, n_g)
            };

        // UV interpolation
        let uv = if let Some(uv_data) = &mesh.uv {
            let d0 = &uv_data[index.x];
            let d1 = &uv_data[index.y];
            let d2 = &uv_data[index.z];
            Some(d0 * (1.0 - hit_u - hit_v) + d1 * hit_u + d2 * hit_v)
        } else {
            None
        };

        let frame = Frame::new(n_s);
        let wi = frame.to_local(-ray.d);
        Intersection {
            dist,
            n_g,
            n_s,
            p,
            uv,
            mesh,
            frame,
            wi,
        }
    }
}

#[derive(Clone, Debug)]
pub struct VarianceEstimator {
    pub mean: f32,
    pub mean_sqr: f32,
    pub sample_count: u32,
}
impl VarianceEstimator {
    fn add(&mut self, v: f32) {
        self.sample_count += 1;
        let delta = v - self.mean;
        self.mean += delta / self.sample_count as f32;
        self.mean_sqr += delta * (v - self.mean);
    }

    fn variance(&self) -> f32 {
        self.mean_sqr / (self.sample_count - 1) as f32
    }
}
impl Default for VarianceEstimator {
    fn default() -> Self {
        Self {
            mean: 0.0,
            mean_sqr: 0.0,
            sample_count: 0,
        }
    }
}
