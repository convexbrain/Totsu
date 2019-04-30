use totsu::prelude::*;
use totsu::predef::QCQP;

use std::io::prelude::*;
use std::io::BufWriter;
use std::fs::File;

use std::str::FromStr;

/// main
fn main() -> std::io::Result<()> {

    //----- parameters

    let t_cap = 30; // # of grids dividing time [0, 1)
    let mut a_cap = 90.0; // limit acceleration magnitude
    let x_s = (0., 0.); // start position
    let x_m1 = (0.5, -1.5); // target position at time=1/3
    let x_m2 = (0.25, 1.5); // target position at time=2/3
    let x_t = (1., 1.); // terminal position

    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 {
        if let Ok(a) = FP::from_str(&args[1]) {
            a_cap = a; // a_cap can be specified by 1st argument
        }
    }

    //----- formulate path-planning as QCQP

    let n = t_cap * 2;
    let m = t_cap - 2;
    let p = 6 * 2;

    let mut mat_p = vec![Mat::new(n, n); m + 1];
    let vec_q = vec![Mat::new_vec(n); m + 1];
    let mut scl_r = vec![0. as FP; m + 1];

    let dt = 1. / t_cap as FP;
    let mut mat_d = Mat::new(n, n);
    for i in 0 .. t_cap - 1 {
        let ti = i;
        mat_d.put(ti, ti, -1. / dt);
        mat_d.put(ti, ti + 1, 1. / dt);
        let ti = t_cap + i;
        mat_d.put(ti, ti, -1. / dt);
        mat_d.put(ti, ti + 1, 1. / dt);
    }
    // minimize total squared magnitude of velocity
    mat_p[0].assign(&(mat_d.t() * &mat_d));

    let dtdt = dt * dt;
    for i in 0 .. t_cap - 2 {
        let mut mat_d = Mat::new(n, n);
        let mi = i + 1;
        let ti = i;
        mat_d.put(ti, ti, 1. / dtdt);
        mat_d.put(ti, ti + 1, -2. / dtdt);
        mat_d.put(ti, ti + 2, 1. / dtdt);
        let ti = t_cap + i;
        mat_d.put(ti, ti, 1. / dtdt);
        mat_d.put(ti, ti + 1, -2. / dtdt);
        mat_d.put(ti, ti + 2, 1. / dtdt);

        // limit acceleration magnitude
        mat_p[mi].assign(&(mat_d.t() * &mat_d));
        scl_r[mi] = -0.5 * a_cap * a_cap;
    }

    let mut mat_a = Mat::new(p, n);
    let mut vec_b = Mat::new_vec(p);

    // target point: x(0) = x_s, v(0) = 0
    // x0
    mat_a.put(0, 0, 1.);
    vec_b.put(0, 0, x_s.0);
    // x1
    mat_a.put(1, t_cap, 1.);
    vec_b.put(1, 0, x_s.0);
    // v0
    mat_a.put(2, 0, -1.);
    mat_a.put(2, 1, 1.);
    // v1
    mat_a.put(3, t_cap, -1.);
    mat_a.put(3, t_cap + 1, 1.);

    // target point: x(1) = x_t, v(1) = 0
    // x0
    mat_a.put(4, t_cap - 1, 1.);
    vec_b.put(4, 0, x_t.0);
    // x1
    mat_a.put(5, t_cap * 2 - 1, 1.);
    vec_b.put(5, 0, x_t.1);
    // v0
    mat_a.put(6, t_cap - 2, -1.);
    mat_a.put(6, t_cap - 1, 1.);
    // v1
    mat_a.put(7, t_cap * 2 - 2, -1.);
    mat_a.put(7, t_cap * 2 - 1, 1.);

    // target point: x(1/3) = x_m1
    let t_m1 = t_cap / 3;
    // x0
    mat_a.put(8, t_m1, 1.);
    vec_b.put(8, 0, x_m1.0);
    // x1
    mat_a.put(9, t_cap + t_m1, 1.);
    vec_b.put(9, 0, x_m1.1);

    // target point: x(2/3) = x_m2
    let t_m2 = t_cap * 2 / 3;
    // x0
    mat_a.put(10, t_m2, 1.);
    vec_b.put(10, 0, x_m2.0);
    // x1
    mat_a.put(11, t_cap + t_m2, 1.);
    vec_b.put(11, 0, x_m2.1);

    //----- solve QCQP

    let param = PDIPMParam::default();
    let rslt = PDIPM::new().solve_qcqp(&param, &mut std::io::sink(),
                                       &mat_p, &vec_q, &scl_r,
                                       &mat_a, &vec_b).unwrap();
    //println!("{}", rslt);

    //----- file output for graph plot

    let mut dat_target = BufWriter::new(File::create("dat_target")?);
    writeln!(dat_target, "{} {}", x_s.0, x_s.1)?;
    writeln!(dat_target, "{} {}", x_m1.0, x_m1.1)?;
    writeln!(dat_target, "{} {}", x_m2.0, x_m2.1)?;
    writeln!(dat_target, "{} {}", x_t.0, x_t.1)?;

    let mut dat_point = BufWriter::new(File::create("dat_point")?);

    for i in 0 .. t_cap {
        writeln!(dat_point, "{} {} {}", i as FP * dt, rslt.get(i, 0), rslt.get(t_cap + i, 0))?;
    }

    Ok(())
}
