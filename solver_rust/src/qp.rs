use super::mat::{Mat, MatSlice, MatSliMu, FP, FP_EPSILON}; // TODO: prelude

const MARGIN: FP = 1.;
const NLOOP: usize = 256;
const BLOOP: usize = 256;
const EPS_FEAS: FP = FP_EPSILON;
const EPS_ETA: FP = FP_EPSILON;
const MU: FP = 10.;
const ALPHA: FP = 0.1;
const BETA: FP = 0.8;
const S_COEF: FP = 0.99;

pub fn solve<F1, F2, F3, F4, F5, F6, F7>(
    n: usize, m: usize, p: usize,
    initial_point: F1,
    d_objective: F2,
    dd_objective: F3,
    inequality: F4,
    d_inequality: F5,
    dd_inequality: F6,
    equality: F7
) -> Result<Mat, &'static str>
where F1: Fn(MatSliMu),
      F2: Fn(&MatSlice, &mut Mat),
      F3: Fn(),
      F4: Fn(&MatSlice, &mut Mat),
      F5: Fn(&MatSlice, &mut Mat),
      F6: Fn(),
      F7: Fn(&mut Mat, &mut Mat)
{
    // parameter check
    if n == 0 {return Err("n: 0");}

    /***** matrix *****/
    // constant across loop
    let mut mat_a = Mat::new(p, n);
    let mut vec_b = Mat::new_vec(p);
    // loop variable
    let mut vec_y = Mat::new_vec(n + m + p);
    let mut vec_dy = Mat::new_vec(n + m + p);
    let mut kkt = Mat::new(n + m + p, n + m + p);
    // temporal in loop
    let mut vec_df_o = Mat::new_vec(n);
    let mut vec_f_i = Mat::new_vec(m);
    let mut vec_r_t = Mat::new_vec(n + m + p);
    let mut vec_y_p = Mat::new_vec(n + m + p);
    let mut mat_df_i = Mat::new(m, n);
    let mut mat_ddf = Mat::new(n, n);

    {   // initialize
        let x = vec_y.slice_mut(0 .. n, ..);
        initial_point(x);
        let mut lmd = vec_y.slice_mut(n .. n + m, ..);
        lmd.assign_all(MARGIN);
        equality(&mut mat_a, &mut vec_b);
    }
    {   // initial df_o, f_i, df_i
        let x = vec_y.slice(0 .. n, ..);
        d_objective(&x, &mut vec_df_o);
        inequality(&x, &mut vec_f_i);
        d_inequality(&x, &mut mat_df_i);

        // inequality feasibility check
        if vec_f_i.max() >= 0. {return Err("inequality: not feasible with initial value");}
    }
    {   // initial residual - dual and primal
        let mut r_dual = vec_r_t.slice_mut(0 .. n, ..);
        r_dual.assign(&vec_df_o);
        if m > 0 {
            let lmd = vec_y.slice(n .. n + m, ..);
            r_dual.assign(&(&r_dual + mat_df_i.t() * lmd));
        }
        if p > 0 {
            let nu = vec_y.slice(n + m .. n + m + p, ..);
            r_dual.assign(&(&r_dual + mat_a.t() * nu));
        }

        let mut r_pri = vec_r_t.slice_mut(n + m .. n + m + p, ..);
        if p > 0 {
            let x = vec_y.slice(0 .. n, ..);
            r_pri.assign(&(mat_a * x - vec_b));
        }
    }

    for _ in 0 .. NLOOP {

        /***** calc t *****/

        let eta = if m > 0 {
            let lmd = vec_y.slice(n .. n + m, ..);
            -(vec_f_i.t() * lmd)[(0, 0)]
        }
        else {
            EPS_ETA
        };

        // inequality feasibility check
		if eta < 0. {return Err("inequality: not feasible");}

    }

    Err("not implemented")

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

    let s = -vec_h.min();
    let mut margin = MARGIN;
    let mut s_inital = s + margin;
    while s_inital <= s {
        s_inital = s + margin;
        margin *= 2.;
    }

    // ----- start to solve

    let rslt = solve(n + 1, m, p + 1, // '+ 1' is for a slack variable
          |mut x| {
              x[(n, 0)] = s_inital;
          },
          |x, df_o| {
              df_o.slice_mut(0 .. n, ..).assign(
                  &(mat_p * x.slice(0 .. n, ..) + vec_q)
              );
              df_o[(n, 0)] = 0.;
          },
          || {},
          |x, f_i| {
              f_i.assign(
                  &(mat_g * x.slice(0 .. n, ..) - vec_h - x[(n, 0)] * (m as FP))
              )
          },
          |_, df_i| {
              df_i.slice_mut(0 .. m, 0 .. n).assign(&mat_g);
              df_i.slice_mut(0 .. m, n ..= n).assign_all(-(m as FP));
          },
          || {},
          |a, b| {
              a.slice_mut(0 .. p, 0 .. n).assign(mat_a);
              b.slice_mut(0 .. p, ..).assign(vec_b);
              // for a slack variable
              a[(p, n)] = 1.;
          }
    );

    match rslt {
        Ok(x) => Ok(x.slice(0 .. n, ..).clone()),
        Err(s) => Err(s)
    }
}

#[test]
fn test_qp()
{
    const n: usize = 2; // x0, x1
    const m: usize = 1;
    const p: usize = 0;

    let vec_x = Mat::new_vec(n);

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
    rslt.unwrap();
}
