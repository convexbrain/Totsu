use totsu::prelude::*;
use totsu::predef::LP;
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
fn wx(x: &Mat, alpha: &MatSlice, xi: &MatSlice) -> FP
{
    let mut f: FP = 0.;
    for i in 0 .. alpha.size().0 {
        f += alpha.get(i, 0) * kernel(&x.col(i), xi);
    }
    f
}

/// main
fn main() -> std::io::Result<()> {

    //----- make sample points for training

    let mut rng = XOR64::init();
    let l = 20; // # of samples
    let x = Mat::new(2, l).set_by(|_, _| rng.next()); // random 2-dimensional points
    let y = Mat::new_vec(l).set_by(|smp, _| {
        let x0 = (5. * x.get(0, smp)).cos();
        let x1 = (7. * x.get(1, smp)).cos();
        x0 * x1 // wavy-shaped points
    });

    //----- formulate L1-regularized L1-error regression as LP

    let n = l * 3 + 1; // z, alpha, beta, bias
    let m = l * 4;
    let p = 0;

    let lambda = 0.2; // L1-regularization strength

    let mut vec_c = Mat::new_vec(n);
    vec_c.rows_mut(0 .. l).assign_all(1.); // for z, L1-norm error
    vec_c.rows_mut(l * 2 .. l * 3).assign_all(lambda); // for beta, L1-regularization

    let mut mat_g = Mat::new(m, n);
    mat_g.slice_mut(0 .. l, 0 .. l).assign_eye(-1.);
    mat_g.slice_mut(l .. l * 2, 0 .. l).assign_eye(-1.);
    mat_g.slice_mut(0 .. l, l .. l * 2).assign_by(|r, c| {
        Some(kernel(&x.col(r), &x.col(c)))
    });
    mat_g.slice_mut(l .. l * 2, l .. l * 2).assign_by(|r, c| {
        Some(-kernel(&x.col(r), &x.col(c)))
    });
    mat_g.slice_mut(l * 2 .. l * 3, l .. l * 2).assign_eye(1.);
    mat_g.slice_mut(l * 3 .. l * 4, l .. l * 2).assign_eye(-1.);
    mat_g.slice_mut(l * 3 .. l * 4, l * 2 .. l * 3).assign_eye(-1.);
    mat_g.slice_mut(l * 3 .. l * 4, l * 2 .. l * 3).assign_eye(-1.);
    mat_g.slice_mut(0 .. l, l * 3 ..= l * 3).assign_all(1.);
    mat_g.slice_mut(l .. l * 2, l * 3 ..= l * 3).assign_all(-1.);

    let mut vec_h = Mat::new_vec(m);
    vec_h.rows_mut(0 .. l).assign(&y);
    vec_h.rows_mut(l .. l * 2).assign(&-&y);

    let mat_a = Mat::new(p, n);
    let vec_b = Mat::new_vec(p);

    //----- solve LP

    let param = PDIPMParam::default();
    let rslt = PDIPM::new().solve_lp(&param, &mut std::io::sink(),
                                     &vec_c,
                                     &mat_g, &vec_h,
                                     &mat_a, &vec_b).unwrap();
    //println!("{}", rslt);
    let alpha = rslt.rows(l .. l * 2);
    let bias = rslt.get(l * 3, 0);

    //----- file output for graph plot

    let mut dat_point = BufWriter::new(File::create("dat_point")?);

    for smp in 0 .. l {
        writeln!(dat_point, "{} {} {} {}", x.get(0, smp), x.get(1, smp), y.get(smp, 0), alpha.get(smp, 0))?;
    }

    let mut dat_grid = BufWriter::new(File::create("dat_grid")?);

    let grid = 20;
    for iy in 0 .. grid {
        for ix in 0 .. grid {
            let xi = Mat::new_vec(2).set_iter(&[
                ix as FP / (grid - 1) as FP,
                iy as FP / (grid - 1) as FP
            ]);

            let f = wx(&x, &alpha, &xi.as_slice()) + bias;

            write!(dat_grid, " {}", f)?;
        }
        writeln!(dat_grid)?;
    }

    Ok(())
}
