use super::mat::{Mat, MatSlice, MatSliMu, FP, FP_MINPOS, FP_EPSILON}; // TODO: prelude
use super::matsvd::MatSVD; // TODO: prelude

use std::io::Write;

macro_rules! writeln_or {
    ( $( $arg: expr ),* ) => {
        writeln!( $( $arg ),* ).or(Err("log: I/O Error"))
    };
}

const MARGIN: FP = 1.;
const NLOOP: usize = 256;
const BLOOP: usize = 256;
const EPS_FEAS: FP = FP_EPSILON;
const EPS_ETA: FP = FP_EPSILON;
const MU: FP = 10.;
const ALPHA: FP = 0.1;
const BETA: FP = 0.8;
const S_COEF: FP = 0.99;

// TODO: log, debug, module, optimize

pub fn solve<L, Fo1, Fo2, Fi0, Fi1, Fi2, Fe, Fs>(
    n: usize, m: usize, p: usize,
    mut log: L,
    d_objective: Fo1,
    dd_objective: Fo2,
    inequality: Fi0,
    d_inequality: Fi1,
    dd_inequality: Fi2,
    equality: Fe,
    start_point: Fs
) -> Result<Mat, &'static str>
where L: Write,
      Fo1: Fn(&MatSlice, &mut Mat),
      Fo2: Fn(&MatSlice, &mut Mat),
      Fi0: Fn(&MatSlice, &mut Mat),
      Fi1: Fn(&MatSlice, &mut Mat),
      Fi2: Fn(&MatSlice, &mut Mat, usize),
      Fe: FnOnce(&mut Mat, &mut Mat),
      Fs: FnOnce(MatSliMu)
{
    // parameter check
    if n == 0 {return Err("n: 0");}

    /***** matrix *****/
    // constant across loop
    let mut mat_a = Mat::new(p, n);
    let mut vec_b = Mat::new_vec(p);
    // loop variable
    let mut vec_y = Mat::new_vec(n + m + p);
    let mut kkt = Mat::new(n + m + p, n + m + p);
    // temporal in loop
    let mut vec_df_o = Mat::new_vec(n);
    let mut vec_f_i = Mat::new_vec(m);
    let mut vec_r_t = Mat::new_vec(n + m + p);
    let mut mat_df_i = Mat::new(m, n);
    let mut mat_ddf = Mat::new(n, n);

    /***** KKT matrix decomposition solver *****/
    let mut svd = MatSVD::new(kkt.size());
    
    // initialize
    let x = vec_y.rows_mut(0 .. n);
    start_point(x);
    let mut lmd = vec_y.rows_mut(n .. n + m);
    lmd.assign_all(MARGIN);
    equality(&mut mat_a, &mut vec_b);

    // initial df_o, f_i, df_i
    let x = vec_y.rows(0 .. n);
    d_objective(&x, &mut vec_df_o);
    inequality(&x, &mut vec_f_i);
    d_inequality(&x, &mut mat_df_i);

    // inequality feasibility check
    if vec_f_i.max().2 >= 0. {return Err("inequality: not feasible at init");}

    // initial residual - dual and primal
    let mut r_dual = vec_r_t.rows_mut(0 .. n);
    r_dual.assign(&vec_df_o);
    if m > 0 {
        let lmd = vec_y.rows(n .. n + m);
        r_dual.assign(&(&r_dual + mat_df_i.t() * lmd));
    }
    if p > 0 {
        let nu = vec_y.rows(n + m .. n + m + p);
        r_dual.assign(&(&r_dual + mat_a.t() * nu));
    }
    let mut r_pri = vec_r_t.rows_mut(n + m .. n + m + p);
    if p > 0 {
        let x = vec_y.rows(0 .. n);
        r_pri.assign(&(&mat_a * x - &vec_b));
    }

    //

    let mut cnt = 0;
    while cnt < NLOOP {
        writeln_or!(log)?;
        writeln_or!(log, "===== ===== ===== ===== loop : {}", cnt)?;

        let x = vec_y.rows(0 .. n);
        let lmd = vec_y.rows(n .. n + m);

        /***** calc t *****/

        let eta = if m > 0 {
            -(vec_f_i.t() * &lmd)[(0, 0)]
        }
        else {
            EPS_ETA
        };

        // inequality feasibility check
        if eta < 0. {return Err("inequality: not feasible in loop");}

        let m_inv_t = eta / MU;

        /***** update residual - central *****/

        if m > 0 {
            let mut r_cent = vec_r_t.rows_mut(n .. n + m);
            r_cent.assign(&(-lmd.clone_diag() * &vec_f_i - m_inv_t));
        }

        /***** termination criteria *****/

        let r_dual = vec_r_t.rows(0 .. n);
        let r_pri = vec_r_t.rows(n + m .. n + m + p);

        let r_dual_norm = r_dual.norm_p2();
        let r_pri_norm = r_pri.norm_p2();

        writeln_or!(log, "|| r_dual || : {:.3e}", r_dual_norm)?;
        writeln_or!(log, "|| r_pri  || : {:.3e}", r_pri_norm)?;
        writeln_or!(log, "   eta       : {:.3e}", eta)?;

        if (r_dual_norm <= EPS_FEAS) && (r_pri_norm <= EPS_FEAS) && (eta <= EPS_ETA) {
            writeln_or!(log, "termination criteria satisfied")?;
            break;
        }

        /***** calc kkt matrix *****/
        
        let mut kkt_x_dual = kkt.slice_mut(0 .. n, 0 .. n);
        dd_objective(&x, &mut mat_ddf);
        kkt_x_dual.assign(&mat_ddf);
        for i in 0 .. m {
            dd_inequality(&x, &mut mat_ddf, i);
        }

        if m > 0 {
            let mut kkt_lmd_dual = kkt.slice_mut(0 .. n, n .. n + m);
            kkt_lmd_dual.assign(&mat_df_i.t());

            let mut kkt_x_cent = kkt.slice_mut(n .. n + m, 0 .. n);
            kkt_x_cent.assign(&(-lmd.clone_diag() * &mat_df_i));

            let mut kkt_lmd_cent = kkt.slice_mut(n .. n + m, n .. n + m);
            kkt_lmd_cent.assign(&(-vec_f_i.clone_diag()));
        }

        if p > 0 {
            let mut kkt_nu_dual = kkt.slice_mut(0 .. n, n + m .. n + m + p);
            kkt_nu_dual.assign(&mat_a.t());

            let mut kkt_x_pri = kkt.slice_mut(n + m .. n + m + p, 0 .. n);
            kkt_x_pri.assign(&mat_a);
        }

        /***** calc search direction *****/

        //svd.decomp(&kkt);
        svd.decomp_warm(&kkt);
        
        let dy = svd.solve(&(-&vec_r_t));

        writeln_or!(log, "y : {}", vec_y.t())?;
        //writeln_or!(log, "kkt : {}", kkt)?;
        writeln_or!(log, "r_t : {}", vec_r_t.t())?;
        writeln_or!(log, "dy : {}", dy.t())?;

        /***** back tracking line search - from here *****/

        let mut s_max: FP = 1.;
        {
            let dlmd = dy.rows(n .. n + m);

            for i in 0 .. m {
                if dlmd[(i, 0)] < -FP_MINPOS { // to avoid zero-division by Dlmd
                    s_max = s_max.min(-lmd[(i, 0)] / dlmd[(i, 0)]);
                }
            }
        }
        let mut s = S_COEF * s_max;

        let mut y_p = &vec_y + s * &dy;

        let mut bcnt = 0;
        while bcnt < BLOOP {
            let x_p = y_p.rows(0 .. n);
            let lmd_p = y_p.rows(n .. n + m);
            
            // update f_i
            inequality(&x_p, &mut vec_f_i);

            if (vec_f_i.max().2 < 0.) && (lmd_p.min().2 > 0.) {break;}
            s = BETA * s;
            y_p = &vec_y + s * &dy;

            bcnt += 1;
        }

        writeln_or!(log, "s : {:.3e}", s)?;

        if bcnt < BLOOP {
            writeln_or!(log, "feasible points found")?;
        }
        else {
            writeln_or!(log, "infeasible in this direction")?;
        }

        let org_r_t_norm = vec_r_t.norm_p2();

        while bcnt < BLOOP {
            let x_p = y_p.rows(0 .. n);
            let lmd_p = y_p.rows(n .. n + m);
            let nu_p = y_p.rows(n + m .. n + m + p);

            // update df_o, f_i, df_i
            d_objective(&x_p, &mut vec_df_o);
            inequality(&x_p, &mut vec_f_i);
            d_inequality(&x_p, &mut mat_df_i);

            // update residual
            let mut r_dual = vec_r_t.rows_mut(0 .. n);
            r_dual.assign(&vec_df_o);
            if m > 0 {
                r_dual.assign(&(&r_dual + mat_df_i.t() * &lmd_p));
            }
            if p > 0 {
                r_dual.assign(&(&r_dual + mat_a.t() * nu_p));
            }
            if m > 0 {
                let mut r_cent = vec_r_t.rows_mut(n .. n + m);
                r_cent.assign(&(-lmd_p.clone_diag() * &vec_f_i - m_inv_t));
            }
            if p > 0 {
                let mut r_pri = vec_r_t.rows_mut(n + m .. n + m + p);
                r_pri.assign(&(&mat_a * x_p - &vec_b));
            }

            if vec_r_t.norm_p2() <= (1. - ALPHA * s) * org_r_t_norm {break;}
            s = BETA * s;
            y_p = &vec_y + s * &dy;

            bcnt += 1;
        }

        writeln_or!(log, "s : {:.3e}", s)?;

        if (bcnt < BLOOP) && ((&y_p - &vec_y).norm_p2() >= FP_EPSILON) {
            writeln_or!(log, "update")?;
            // update y
            vec_y.assign(&y_p);
        }
        else {
            writeln_or!(log, "no more improvement")?;
            return Err("line search: not converged");
        }

        /***** back tracking line search - to here *****/

        cnt += 1;
    }

    if !(cnt < NLOOP) {
        writeln_or!(log, "iteration limit")?;
        return Err("iteration: not converged");
    }

    writeln_or!(log)?;
    writeln_or!(log, "===== ===== ===== ===== result")?;
    let x = vec_y.rows(0 .. n);
    let lmd = vec_y.rows(n .. n + m);
    let nu = vec_y.rows(n + m .. n + m + p);
    writeln_or!(log, "x : {}", x.t())?;
    writeln_or!(log, "lmd : {}", lmd.t())?;
    writeln_or!(log, "nu : {}", nu.t())?;

    Ok(vec_y)
}

pub fn solve_qp(mat_p: &Mat, vec_q: &Mat,
                mat_g: &Mat, vec_h: &Mat,
                mat_a: &Mat, vec_b: &Mat)
                -> Result<Mat, &'static str>
{
    // ----- parameter check

    let (n, tmpc) = mat_p.size();

    if n == 0 {return Err("mat_p: 0 rows");}
    if tmpc != n {return Err("mat_p: must be square matrix");}

    let (tmpr, tmpc) = vec_q.size();
    
    if tmpc != 1 {return Err("vec_q: must be column vector");}
    if tmpr != n {return Err("vec_q: size mismatch");}

    // m = 0 means NO inequality constraints
    let (m, tmpc) = mat_g.size();

    if tmpc != n {return Err("mat_g: column size mismatch");}

    let (tmpr, tmpc) = vec_h.size();
    
    if tmpc != 1 {return Err("vec_h: must be column vector");}
    if tmpr != m {return Err("vec_h: size mismatch");}

    // p = 0 means NO equality constraints
    let (p, tmpc) = mat_a.size();

    if tmpc != n {return Err("mat_a: column size mismatch");}

    let (tmpr, tmpc) = vec_b.size();
    
    if tmpc != 1 {return Err("vec_b: must be column vector");}
    if tmpr != p {return Err("vec_b: size mismatch");}

    // ----- initial value of a slack variable

    let s = -vec_h.min().2;
    let mut margin = MARGIN;
    let mut s_inital = s + margin;
    while s_inital <= s {
        s_inital = s + margin;
        margin *= 2.;
    }

    // ----- start to solve

    let rslt = solve(n + 1, m, p + 1, // '+ 1' is for a slack variable
        std::io::sink(),
        |x, df_o| {
            df_o.rows_mut(0 .. n).assign(
                &(mat_p * x.rows(0 .. n) + vec_q)
            );
            df_o[(n, 0)] = 0.;
        },
        |_, ddf_o| {
            ddf_o.slice_mut(0 .. n, 0 .. n).assign(mat_p);
            ddf_o.row_mut(n).assign_all(0.);
            ddf_o.col_mut(n).assign_all(0.);
        },
        |x, f_i| {
            f_i.assign(
                &(mat_g * x.rows(0 .. n) - vec_h - x[(n, 0)] * (m as FP))
            )
        },
        |_, df_i| {
            df_i.cols_mut(0 .. n).assign(&mat_g);
            df_i.col_mut(n).assign_all(-(m as FP));
        },
        |_, ddf_i, _| {
            ddf_i.assign_all(0.);
        },
        |a, b| {
            a.slice_mut(0 .. p, 0 .. n).assign(mat_a);
            b.rows_mut(0 .. p).assign(vec_b);
            // for a slack variable
            a[(p, n)] = 1.;
        },
        |mut x| {
            x[(n, 0)] = s_inital;
        }
    );

    match rslt {
        Ok(y) => Ok(y.rows(0 .. n).clone()),
        Err(s) => Err(s)
    }
}

#[test]
fn test_qp()
{
    let n: usize = 2; // x0, x1
    let m: usize = 1;
    let p: usize = 0;

    // (1/2)(x - a)^2 + const
    let mat_p = Mat::new(n, n).set_iter(&[
        1., 0.,
        0., 1.
    ]);
    let vec_q = Mat::new_vec(n).set_iter(&[
        -(-1.), // -a0
        -(-2.)  // -a1
    ]);

    // 1 - x0/b0 - x1/b1 <= 0
    let mat_g = Mat::new(m, n).set_iter(&[
        -1. / 2., // -1/b0
        -1. / 3.  // -1/b1
    ]);
    let vec_h = Mat::new_vec(m).set_iter(&[
        -1.
    ]);

    let mat_a = Mat::new(p, n);
    let vec_b = Mat::new_vec(p);

    let rslt = solve_qp(&mat_p, &vec_q,
                        &mat_g, &vec_h,
                        &mat_a, &vec_b);
    println!("{}", rslt.unwrap());
}
