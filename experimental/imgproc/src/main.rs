use totsu::prelude::*;
use totsu::linalg::F64LAPACK;

use utils;

use image::io::Reader as ImageReader;
use image::{Pixel, ImageBuffer, Luma};
use intel_mkl_src as _;
use anyhow::{Result, Context};

type ASolver = Solver<F64LAPACK, f64>;

fn main() -> Result<()> {
    env_logger::init();

    //----- read input image

    let in_img_file = std::env::args().nth(1).context("no input image filename")?;
    println!("{}", in_img_file);
    let in_img = ImageReader::open(in_img_file)?.decode()?;

    let in_mono = in_img.to_luma8();

    let width = in_mono.width();
    let height = in_mono.height();
    let vec_xh: Vec<f64> = in_mono.pixels()
                           .flat_map(|p| p.channels())
                           .map(|p| {*p as f64 / 255.0})
                           .collect();
    println!("{}x{}={}", width, height, vec_xh.len());

    //----- solve user-defined problem

    let _ = ASolver::new().par(|p| {
        p.eps_acc = 1e-3;
        utils::set_par_by_env(p);
    });
    //let rslt = s.solve(...)?;
    //println!("{:?}", rslt);

    //----- write output image

    let vec_x = vec_xh;
    let out_mono = ImageBuffer::from_fn(width, height, |x, y| {
        let v = vec_x[(x + y * width) as usize];
        let v = if v < 0. {0.} else {v};
        let v = if v > 1. {1.} else {v};
        let v = (v * 255.) as u8;
        Luma::from([v])
    });
    out_mono.save("out.png")?;

    Ok(())
}
