use totsu::prelude::*;
use totsu::predef::QCQP;

use std::io::prelude::*;
use std::io::BufWriter;
use std::fs::File;

/// main
fn main() -> std::io::Result<()> {

    let t = 30;
    let acc_max = 0.1;
    let x_m1 = (0.5, -1.5);
    let x_m2 = (0.25, 1.5);
    let x_e = (1., 1.);

    let n = t * 2;
    let m = t - 2;
    let p = 6 * 2;

    let mut mat_p = vec![Mat::new(n, n); m + 1];
    let vec_q = vec![Mat::new_vec(n); m + 1];
    let mut scl_r = vec![0. as FP; m + 1];

    let mut mat_d = Mat::new(n, n);
    for i in 0 .. t - 1 {
        let ti = i;
        mat_d[(ti, ti)] = -1.;
        mat_d[(ti, ti + 1)] = 1.;
        let ti = t + i;
        mat_d[(ti, ti)] = -1.;
        mat_d[(ti, ti + 1)] = 1.;
    }
    mat_p[0].assign(&(mat_d.t() * &mat_d));

    for i in 0 .. t - 2 {
        let mut mat_d = Mat::new(n, n);
        let mi = i + 1;
        let ti = i;
        mat_d[(ti, ti)] = 1.;
        mat_d[(ti, ti + 1)] = -2.;
        mat_d[(ti, ti + 2)] = 1.;
        let ti = t + i;
        mat_d[(ti, ti)] = 1.;
        mat_d[(ti, ti + 1)] = -2.;
        mat_d[(ti, ti + 2)] = 1.;

        mat_p[mi].assign(&(mat_d.t() * &mat_d));

        scl_r[mi] = -0.5 * acc_max * acc_max;
    }

    let mut mat_a = Mat::new(p, n);
    let mut vec_b = Mat::new_vec(p);

    // x(0) = 0, v(0) = 0
    // x0
    mat_a[(0, 0)] = 1.;
    // x1
    mat_a[(1, t)] = 1.;
    // v0
    mat_a[(2, 0)] = -1.;
    mat_a[(2, 1)] = 1.;
    // v1
    mat_a[(3, t)] = -1.;
    mat_a[(3, t + 1)] = 1.;

    // x(t - 1) = x_e, v(t - 1) = 0
    // x0
    mat_a[(4, t - 1)] = 1.;
    vec_b[(4, 0)] = x_e.0;
    // x1
    mat_a[(5, t * 2 - 1)] = 1.;
    vec_b[(5, 0)] = x_e.1;
    // v0
    mat_a[(6, t - 2)] = -1.;
    mat_a[(6, t - 1)] = 1.;
    // v1
    mat_a[(7, t * 2 - 2)] = -1.;
    mat_a[(7, t * 2 - 1)] = 1.;

    // x(t_m1) = x_m1
    let t_m1 = t / 3;
    // x0
    mat_a[(8, t_m1)] = 1.;
    vec_b[(8, 0)] = x_m1.0;
    // x1
    mat_a[(9, t + t_m1)] = 1.;
    vec_b[(9, 0)] = x_m1.1;

    // x(t_m2) = x_m2
    let t_m2 = t * 2 / 3;
    // x0
    mat_a[(10, t_m2)] = 1.;
    vec_b[(10, 0)] = x_m2.0;
    // x1
    mat_a[(11, t + t_m2)] = 1.;
    vec_b[(11, 0)] = x_m2.1;

    //----- solve QCQP

    let param = PDIPMParam::default();
    let rslt = PDIPM::new().solve_qcqp(&param, &mut std::io::sink(),
                                       &mat_p, &vec_q, &scl_r,
                                       &mat_a, &vec_b).unwrap();
    //println!("{}", rslt);

    //----- file output for graph plot

    let mut dat_point = BufWriter::new(File::create("dat_point")?);

    for i in 0 .. t {
        writeln!(dat_point, "{} {}", rslt[(i, 0)], rslt[(t + i, 0)])?;
    }

    Ok(())
}
