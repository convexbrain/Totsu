use totsu::prelude::*;
use totsu::operator::MatBuild;
use totsu::linalg::F64LAPACK;
use totsu::problem::ProbQCQP;

use std::io::prelude::*;
use std::io::BufWriter;
use std::fs::File;
use std::str::FromStr;

extern crate intel_mkl_src;

type AMatBuild = MatBuild<F64LAPACK, f64>;
type AProbQCQP = ProbQCQP<F64LAPACK, f64>;
type ASolver = Solver<F64LAPACK, f64>;

/// main
fn main() -> std::io::Result<()> {
    env_logger::init();

    //----- parameters

    let t_cap = 30; // # of grids dividing time [0, 1)
    let mut a_cap = 90.0; // limit acceleration magnitude
    let x_s = (0., 0.); // start position
    let x_m1 = (0.5, -1.5); // target position at time=1/3
    let x_m2 = (0.25, 1.5); // target position at time=2/3
    let x_t = (1., 1.); // terminal position

    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 {
        if let Ok(a) = f64::from_str(&args[1]) {
            a_cap = a; // a_cap can be specified by 1st argument
        }
    }

    //----- formulate path-planning as QCQP

    let n = t_cap * 2;
    let m = t_cap - 2;
    let p = 6 * 2;

    let mut syms_p = vec![AMatBuild::new(MatType::SymPack(n)); m + 1];
    let vecs_q = vec![AMatBuild::new(MatType::General(n, 1)); m + 1];
    let mut scls_r = vec![0.; m + 1];

    let dt = 1. / t_cap as f64;
    let mut mat_d = AMatBuild::new(MatType::General(n, n));
    for i in 0 .. t_cap - 1 {
        let ti = i;
        mat_d[(ti, ti)] = -1. / dt;
        mat_d[(ti, ti + 1)] = 1. / dt;
        let ti = t_cap + i;
        mat_d[(ti, ti)] = -1. / dt;
        mat_d[(ti, ti + 1)] = 1. / dt;
    }
    // minimize total squared magnitude of velocity
    syms_p[0].set_by_fn(|r, c| {
        let mut v = 0.;
        for i in 0.. n {
            v += mat_d[(i, r)] * mat_d[(i, c)];
        }
        v
    });

    let dtdt = dt * dt;
    for i in 0 .. t_cap - 2 {
        let mut mat_d = AMatBuild::new(MatType::General(n, n));
        let mi = i + 1;
        let ti = i;
        mat_d[(ti, ti)] = 1. / dtdt;
        mat_d[(ti, ti + 1)] = -2. / dtdt;
        mat_d[(ti, ti + 2)] = 1. / dtdt;
        let ti = t_cap + i;
        mat_d[(ti, ti)] = 1. / dtdt;
        mat_d[(ti, ti + 1)] = -2. / dtdt;
        mat_d[(ti, ti + 2)] = 1. / dtdt;

        // limit acceleration magnitude
        syms_p[mi].set_by_fn(|r, c| {
            let mut v = 0.;
            for i in 0.. n {
                v += mat_d[(i, r)] * mat_d[(i, c)];
            }
            v
        });
        scls_r[mi] = -0.5 * a_cap * a_cap;
    }

    let mut mat_a = AMatBuild::new(MatType::General(p, n));
    let mut vec_b = AMatBuild::new(MatType::General(p, 1));

    // target point: x(0) = x_s, v(0) = 0
    // x0
    mat_a[(0, 0)] = 1.;
    vec_b[(0, 0)] = x_s.0;
    // x1
    mat_a[(1, t_cap)] = 1.;
    vec_b[(1, 0)] = x_s.0;
    // v0
    mat_a[(2, 0)] = -1.;
    mat_a[(2, 1)] = 1.;
    // v1
    mat_a[(3, t_cap)] = -1.;
    mat_a[(3, t_cap + 1)] = 1.;

    // target point: x(1) = x_t, v(1) = 0
    // x0
    mat_a[(4, t_cap - 1)] = 1.;
    vec_b[(4, 0)] = x_t.0;
    // x1
    mat_a[(5, t_cap * 2 - 1)] = 1.;
    vec_b[(5, 0)] = x_t.1;
    // v0
    mat_a[(6, t_cap - 2)] = -1.;
    mat_a[(6, t_cap - 1)] = 1.;
    // v1
    mat_a[(7, t_cap * 2 - 2)] = -1.;
    mat_a[(7, t_cap * 2 - 1)] = 1.;

    // target point: x(1/3) = x_m1
    let t_m1 = t_cap / 3;
    // x0
    mat_a[(8, t_m1)] = 1.;
    vec_b[(8, 0)] = x_m1.0;
    // x1
    mat_a[(9, t_cap + t_m1)] = 1.;
    vec_b[(9, 0)] = x_m1.1;

    // target point: x(2/3) = x_m2
    let t_m2 = t_cap * 2 / 3;
    // x0
    mat_a[(10, t_m2)] = 1.;
    vec_b[(10, 0)] = x_m2.0;
    // x1
    mat_a[(11, t_cap + t_m2)] = 1.;
    vec_b[(11, 0)] = x_m2.1;

    //----- solve QCQP

    let s = ASolver::new().par(|p| {
        p.eps_acc = 1e-3;
    });
    let mut qp = AProbQCQP::new(syms_p, vecs_q, scls_r, mat_a, vec_b, s.par.eps_zero);
    let rslt = s.solve(qp.problem()).unwrap();
    //println!("{:?}", rslt);

    //----- file output for graph plot
    // TODO: use some plotting crate

    let mut dat_target = BufWriter::new(File::create("dat_target")?);
    writeln!(dat_target, "{} {}", x_s.0, x_s.1)?;
    writeln!(dat_target, "{} {}", x_m1.0, x_m1.1)?;
    writeln!(dat_target, "{} {}", x_m2.0, x_m2.1)?;
    writeln!(dat_target, "{} {}", x_t.0, x_t.1)?;

    let mut dat_point = BufWriter::new(File::create("dat_point")?);

    for i in 0 .. t_cap {
        writeln!(dat_point, "{} {} {}", i as f64 * dt, rslt.0[i], rslt.0[t_cap + i])?;
    }

    Ok(())
}
