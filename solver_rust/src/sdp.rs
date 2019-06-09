//! Pre-defined SDP solver

use super::prelude::*;
use super::matsvd::{MatSVD, SVDS};
use super::matlinalg;

use std::io::Write;
macro_rules! writeln_or {
    ( $( $arg: expr ),* ) => {
        writeln!( $( $arg ),* ).or(Err("log: I/O Error"))
    };
}

use std::cell::RefCell;

/// Semidefinite program
/// 
/// <script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML' async></script>
/// 
/// The problem is
/// \\[
/// \\begin{array}{ll}
/// {\\rm minimize} & c^Tx \\\\
/// {\\rm subject \\ to} & \\sum_{i=0}^{n - 1} x_i F_i + F_n \\preceq 0 \\\\
/// & A x = b,
/// \\end{array}
/// \\]
/// where
/// - variables \\( x \\in {\\bf R}^n \\)
/// - \\( c \\in {\\bf R}^n \\)
/// - \\( F_j \\in {\\bf S}^k \\) for \\( j = 0, \\ldots, n \\)
/// - \\( A \\in {\\bf R}^{p \\times n} \\), \\( b \\in {\\bf R}^p \\).
/// 
/// Internally a slack variable \\( s \\in {\\bf R} \\) is introduced for the infeasible start method as follows:
/// \\[
/// \\begin{array}{ll}
/// {\\rm minimize} & c^Tx \\\\
/// {\\rm subject \\ to} & \\sum_{i=0}^{n - 1} x_i F_i + F_n \\preceq s I \\\\
/// & A x = b \\\\
/// & s = 0.
/// \\end{array}
/// \\]
pub trait SDP {
    fn solve_sdp<L>(&mut self, param: &PDIPMParam, log: &mut L,
                    vec_c: &Mat, mat_f: &[Mat],
                    mat_a: &Mat, vec_b: &Mat)
                    -> Result<Mat, String>
    where L: Write;
}

fn check_param(vec_c: &Mat, mat_f: &[Mat],
               mat_a: &Mat, vec_b: &Mat)
               -> Result<(usize, usize, usize, usize), String>
{
        let (n, _) = vec_c.size();
        let m = 0;
        let (p, _) = mat_a.size();

        if n == 0 {return Err("vec_c: 0 rows".into());}
        // m = 0 means NO inequality constraints
        // p = 0 means NO equality constraints

        if vec_c.size() != (n, 1) {return Err(format!("vec_c: size {:?} must be {:?}", vec_c.size(), (n, 1)));}

        if mat_f.len() != n + 1 {return Err(format!("mat_f: length {} must be {}", mat_f.len(), n + 1));}

        let (k, _) = mat_f[0].size();
        for i in 0 ..= n {
            if mat_f[i].size() != (k, k) {return Err(format!("mat_f[{}]: size {:?} must be {:?}", i, mat_f[i].size(), (k, k)));}
        }

        if mat_a.size() != (p, n) {return Err(format!("mat_a: size {:?} must be {:?}", mat_a.size(), (p, n)));}
        if vec_b.size() != (p, 1) {return Err(format!("vec_b: size {:?} must be {:?}", vec_b.size(), (p, 1)));}

        Ok((n, m, p, k))
}

impl SDP for PDIPM
{
    /// Runs the solver with given parameters.
    /// 
    /// Returns `Ok` with optimal \\(x\\) or `Err` with message string.
    /// * `param` is solver parameters.
    /// * `log` outputs solver progress.
    /// * `vec_c` is \\(c\\).
    /// * `mat_f` is \\(F_0, \\ldots, F_n\\).
    /// * `mat_a` is \\(A\\).
    /// * `vec_b` is \\(b\\).
    /// 
    /// *NOTE: Current implementation that uses the barrier method is not so accurate.*
    fn solve_sdp<L>(&mut self, param: &PDIPMParam, log: &mut L,
                    vec_c: &Mat, mat_f: &[Mat],
                    mat_a: &Mat, vec_b: &Mat)
                    -> Result<Mat, String>
    where L: Write
    {
        // TODO: improve accuracy

        // ----- parameter check

        let (n, m, p, k) = check_param(vec_c, mat_f, mat_a, vec_b)?;

        // ----- initial value of a slack variable

        let s = matlinalg::dom_eig(&mat_f[n]);
        let mut margin = param.margin;
        let mut s_initial = s + margin;
        while s_initial <= s {
            margin *= 2.;
            s_initial = s + margin;
        }

        // ----- initial value of t for barrier method

        let eye = Mat::new(k, k).set_eye(1.);

        let mut vec_q = Mat::new_vec(n);
        let fx0 = &mat_f[n] - s_initial * &eye;
        for i in 0 .. n {
            vec_q[(i, 0)] = SVDS::new(fx0.size()).solve(&fx0, &mat_f[i]).tr();
        }

        let mut mat_p = Mat::new(n, p + 1);
        mat_p.cols_mut(0 .. p).assign(&mat_a.t());
        mat_p.col_mut(p).assign(&vec_c);

        let mut t = SVDS::new(mat_p.size()).solve(&mat_p, &vec_q)[(p, 0)];
        t = t.max(param.eps);

        // ----- start to solve

        let mut vec_xs = Mat::new_vec(n + 1);
        vec_xs[(n, 0)] = s_initial;

        let svd_cell = RefCell::new(MatSVD::new((k, k)));

        loop {
            writeln_or!(log)?;
            writeln_or!(log, "===== ===== ===== ===== barrier loop")?;
            writeln_or!(log, "t = {:.3e}", t)?;

            let rslt = self.solve(param, log,
                n + 1, m, p + 1, // '+ 1' is for a slack variable
                |x, df_o| {
                    let mut fx = - x[(n, 0)] * &eye;
                    fx += &mat_f[n];
                    for i in 0 .. n {
                        fx += &mat_f[i] * x[(i, 0)];
                    }
                    let mut svd = svd_cell.borrow_mut();
                    svd.decomp(&fx);
                    //
                    for i in 0 .. n {
                        df_o[(i, 0)] = t * vec_c[(i, 0)] - svd.solve(&mat_f[i]).tr();
                    }
                    // for a slack variable
                    df_o[(n, 0)] = svd.solve(&eye).tr();
                },
                |_, ddf_o| {
                    // x won't change because dd_objective is called after d_objective with the same x
                    let svd = svd_cell.borrow();
                    //
                    let feye = svd.solve(&eye);
                    for c in 0 .. n {
                        let fc = svd.solve(&mat_f[c]);
                        for r in 0 .. c {
                            let fr = svd.solve(&mat_f[r]);
                            let v = fr.prod(&fc); // tr(fr*fc)
                            ddf_o[(r, c)] = v;
                            ddf_o[(c, r)] = v;
                        }
                        let v = fc.norm_p2sq(); // tr(fc*fc)
                        ddf_o[(c, c)] = v;
                        // for a slack variable
                        let v = -fc.prod(&feye); // tr(fc*-feye)
                        ddf_o[(n, c)] = v;
                        ddf_o[(c, n)] = v;
                    }
                    // for a slack variable
                    let v = feye.norm_p2sq(); // tr(-feye*-feye)
                    ddf_o[(n, n)] = v;
                },
                |_, f_i| {
                    f_i.assign_all(0.);
                },
                |_, df_i| {
                    df_i.assign_all(0.);
                },
                |_, ddf_i, _| {
                    ddf_i.assign_all(0.);
                },
                |a, b| {
                    a.assign_all(0.);
                    b.assign_all(0.);
                    a.slice_mut(0 .. p, 0 .. n).assign(mat_a);
                    b.rows_mut(0 .. p).assign(vec_b);
                    // for a slack variable
                    a[(p, n)] = 1.;
                },
                |mut x| {
                    x.assign(&vec_xs);
                }
            );

            match rslt {
                Ok(y) => vec_xs.assign(&y.rows(0 .. n + 1)),
                Err(other) => return Err(other.into())
            };

            if k as FP / t < param.eps {
                return Ok(vec_xs.rows(0 .. n).clone_sz());
            }

            t *= param.mu;
        }
    }
}
