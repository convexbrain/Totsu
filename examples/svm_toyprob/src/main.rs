use totsu::prelude::*;
use totsu::predef::QP;
use totsu::mat::XOR64;

use std::io::prelude::*;
use std::io::BufWriter;
use std::fs::File;

/// gaussian kernel
fn kernel(xi: &MatSlice, xj: &MatSlice) -> FP
{
        let sigma_sq = 1. / 8.;
        let norm_sq = (xi - xj).norm_p2sq();
        (-norm_sq / sigma_sq).exp()
}

/// kerneled w*x
fn wx(x: &Mat, y: &Mat, alpha: &Mat, xi: &MatSlice) -> FP
{
    let mut f: FP = 0.;
    for i in 0 .. y.size().0 {
        f += alpha.get(i, 0) * y.get(i, 0) * kernel(&x.col(i), xi);
    }
    f
}

/// main
fn main() -> std::io::Result<()> {

    //----- make sample points for training

    let mut rng = XOR64::init();
    let l = 50; // # of samples
    let x = Mat::new(2, l).set_by(|_, _| rng.next()); // random 2-dimensional points
    let y = Mat::new_vec(l).set_by(|smp, _| {
        let (d0, d1) = (x.get(0, smp) - 0.5, x.get(1, smp) - 0.5);
        let r2 = d0 * d0 + d1 * d1;
        let r = r2.sqrt();
        if (r > 0.25) && (r < 0.4) {1.} else {-1.} // ring shape, 1 inside, -1 outside and hole
    });

    //----- formulate dual of hard-margin SVM as QP

    let n = l;
    let m = l;
    let p = 1;

    let mat_p = Mat::new(n, n).set_by(|r, c| {
        y.get(r, 0) * y.get(c, 0) * kernel(&x.col(r), &x.col(c))
    });
    let vec_q = Mat::new_vec(n).set_all(-1.);

    let mat_g = Mat::new(m, n).set_eye(-1.);
    let vec_h = Mat::new_vec(m);

    let mat_a = y.t().clone_sz();
    assert_eq!(mat_a.size(), (p, n));
    let vec_b = Mat::new_vec(p);

    //----- solve QP

    let param = PDIPMParam::default();
    let rslt = PDIPM::new().solve_qp(&param, &mut std::io::sink(),
                                     &mat_p, &vec_q,
                                     &mat_g, &vec_h,
                                     &mat_a, &vec_b).unwrap();
    //println!("{}", rslt);

    //----- calculate bias term

    let mut bias: FP = 0.;
    let mut cnt = 0;
    for i in 0 .. l {
        if rslt.get(i, 0) > 1e-4 {
            bias += y.get(i, 0) - wx(&x, &y, &rslt, &x.col(i));
            cnt += 1;
        }
    }
    bias = bias / cnt as FP;

    //----- file output for graph plot

    let mut dat_point = BufWriter::new(File::create("dat_point")?);

    for smp in 0 .. l {
        writeln!(dat_point, "{} {} {} {}", x.get(0, smp), x.get(1, smp), y.get(smp, 0), rslt.get(smp, 0))?;
    }

    let mut dat_grid = BufWriter::new(File::create("dat_grid")?);

    let grid = 20;
    for iy in 0 .. grid {
        for ix in 0 .. grid {
            let xi = Mat::new_vec(2).set_iter(&[
                ix as FP / (grid - 1) as FP,
                iy as FP / (grid - 1) as FP
            ]);

            let f = wx(&x, &y, &rslt, &xi.as_slice()) + bias;

            write!(dat_grid, " {}", f)?;
        }
        writeln!(dat_grid)?;
    }

    Ok(())
}
