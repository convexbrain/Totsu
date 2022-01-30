use totsu::prelude::*;
use totsu::linalg::F64LAPACK;

use utils;

use image::io::Reader as ImageReader;
use image::{Pixel, ImageBuffer, Luma};
use intel_mkl_src as _;
use anyhow::Result;

type LA = F64LAPACK;
type ASolver = Solver<LA, f64>;

mod laplacian;

mod prob_op_c;
use prob_op_c::ProbOpC;

mod prob_op_a;
use prob_op_a::ProbOpA;

mod prob_op_b;
use prob_op_b::ProbOpB;

mod prob_cone;
use prob_cone::ProbCone;

//

fn main() -> Result<()> {
    env_logger::init();

    //----- read input image

    let in_img_file = std::env::args().nth(1).unwrap_or("miku_face_noise.png".to_string());
    log::info!("file: {}", in_img_file);
    let in_img = ImageReader::open(in_img_file)?.decode()?;

    let in_mono = in_img.to_luma8();

    let width = in_mono.width();
    let height = in_mono.height();
    let vec_xh: Vec<f64> = in_mono.pixels()
                           .flat_map(|p| p.channels())
                           .map(|p| {*p as f64 / 255.0})
                           .collect();
    log::info!("size: {}x{}={}", width, height, vec_xh.len());

    //-----

    let ratio = utils::num_by_env("RATIO").unwrap_or(0.01);

    //----- solve user-defined problem

    let s = ASolver::new().par(|p| {
        p.eps_acc = 1.0 / 256.0;
        utils::set_par_by_env(p);
    });
    let w = width as usize;
    let h = height as usize;
    let op_c = ProbOpC::new(w, h);
    let op_a = ProbOpA::new(w, h);
    let op_b = ProbOpB::new(w, h, ratio, &vec_xh);
    let cone = ProbCone::new(w, h);
    let mut work = vec![0.0; ASolver::query_worklen(op_a.size())];
    let rslt = s.solve((op_c, op_a, op_b, cone, &mut work))?;
    //println!("{:?}", rslt);

    //----- write output image

    let vec_x = rslt.0.split_at((width * height) as usize).0;
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
