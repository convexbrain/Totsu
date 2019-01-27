//! Semidefinite program

use super::prelude::*;
use super::matsvd::MatSVD;

use std::io::Write;
macro_rules! writeln_or {
    ( $( $arg: expr ),* ) => {
        writeln!( $( $arg ),* ).or(Err("log: I/O Error"))
    };
}

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
    fn solve_sdp<L>(&self, log: &mut L,
                    vec_c: &Mat, mat_f: &[Mat],
                    mat_a: &Mat, vec_b: &Mat)
                    -> Result<Mat, &'static str>
    where L: Write;
}

fn check_param(vec_c: &Mat, mat_f: &[Mat],
               mat_a: &Mat, vec_b: &Mat)
               -> Result<(usize, usize, usize, usize), &'static str>
{
        let (n, _) = vec_c.size();
        let m = 0;
        let (p, _) = mat_a.size();

        if n == 0 {return Err("vec_c: 0 rows");}
        // m = 0 means NO inequality constraints
        // p = 0 means NO equality constraints

        if vec_c.size() != (n, 1) {return Err("vec_c: size mismatch");}

        if mat_f.len() != n + 1 {return Err("mat_f: length mismatch");}

        let (k, _) = mat_f[0].size();
        for i in 0 ..= n {
            if mat_f[i].size() != (k, k) {return Err("mat_f[_]: size mismatch");}
        }

        if mat_a.size() != (p, n) {return Err("mat_a: size mismatch");}
        if vec_b.size() != (p, 1) {return Err("vec_b: size mismatch");}

        Ok((n, m, p, k))
}

impl SDP for PDIPM
{
    /// Runs the solver with given parameters.
    /// 
    /// Returns `Ok` with optimal \\(x\\) or `Err` with message string.
    /// * `log` outputs solver progress.
    /// * `vec_c` is \\(c\\).
    /// * `mat_f` is \\(F_0, \\ldots, F_n\\).
    /// * `mat_a` is \\(A\\).
    /// * `vec_b` is \\(b\\).
    /// 
    /// **NOTE: Current implementation is neither efficient nor accurate.**
    /// **You may need increase [`eps`](../pdipm/struct.PDIPM.html#structfield.eps) parameter.**
    fn solve_sdp<L>(&self, log: &mut L,
                    vec_c: &Mat, mat_f: &[Mat],
                    mat_a: &Mat, vec_b: &Mat)
                    -> Result<Mat, &'static str>
    where L: Write
    {
        // TODO: improve efficiency
        // TODO: improve accuracy

        // ----- parameter check

        let (n, m, p, k) = check_param(vec_c, mat_f, mat_a, vec_b)?;

        // ----- initial value of a slack variable

        let mut svd = MatSVD::new((k, k));
        svd.decomp(&mat_f[n]);

        let s = svd.s().max().unwrap();
        let mut margin = self.margin;
        let mut s_initial = s + margin;
        while s_initial <= s {
            margin *= 2.;
            s_initial = s + margin;
        }

        // ----- initial value of t for barrier method

        let eye = Mat::new(k, k).set_eye();

        let mut vec_q = Mat::new_vec(n);
        let fx0 = &mat_f[n] - s_initial * &eye;
        svd.decomp(&fx0); // re-use because of the same size
        for i in 0 .. n {
            vec_q[(i, 0)] = svd.solve(&mat_f[i]).tr();
        }

        let mut mat_p = Mat::new(n, p + 1);
        mat_p.cols_mut(0 .. p).assign(&mat_a.t());
        mat_p.col_mut(p).assign(&vec_c);

        let mut svd = MatSVD::new(mat_p.size());
        svd.decomp(&mat_p);

        let mut t = svd.solve(&vec_q)[(p, 0)];
        t = t.max(self.eps);

        // ----- start to solve

        let mut vec_xs = Mat::new_vec(n + 1);
        vec_xs[(n, 0)] = s_initial;

        while k as FP / t >= self.eps {
            writeln_or!(log)?;
            writeln_or!(log, "===== ===== ===== ===== barrier loop")?;
            writeln_or!(log, "t = {}", t)?;

            let rslt = self.solve(n + 1, m, p + 1, // '+ 1' is for a slack variable
                log,
                |x, df_o| {
                    let mut fx = - x[(n, 0)] * &eye;
                    fx += &mat_f[n];
                    for i in 0 .. n {
                        fx += &mat_f[i] * x[(i, 0)];
                    }
                    let mut svd = MatSVD::new(fx.size());
                    svd.decomp(&fx);
                    //
                    for i in 0 .. n {
                        df_o[(i, 0)] = t * vec_c[(i, 0)] - svd.solve(&mat_f[i]).tr();
                    }
                    // for a slack variable
                    df_o[(n, 0)] = svd.solve(&eye).tr();
                },
                |x, ddf_o| {
                    let mut fx = - x[(n, 0)] * &eye;
                    fx += &mat_f[n];
                    for i in 0 .. n {
                        fx += &mat_f[i] * x[(i, 0)];
                    }
                    let mut svd = MatSVD::new(fx.size());
                    svd.decomp(&fx);
                    //
                    let feye = svd.solve(&eye);
                    for c in 0 .. n {
                        let fc = svd.solve(&mat_f[c]);
                        for r in 0 .. c {
                            let v = (&fc * svd.solve(&mat_f[r])).tr();
                            ddf_o[(r, c)] = v;
                            ddf_o[(c, r)] = v;
                        }
                        let v = (&fc * &fc).tr();
                        ddf_o[(c, c)] = v;
                        // for a slack variable
                        let v = -(&fc * &feye).tr();
                        ddf_o[(n, c)] = v;
                        ddf_o[(c, n)] = v;
                    }
                    // for a slack variable
                    let v = (&feye * &feye).tr();
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
                    a.slice_mut(0 .. p, 0 .. n).assign(mat_a);
                    b.rows_mut(0 .. p).assign(vec_b);
                    // for a slack variable
                    a[(p, n)] = 1.;
                },
                |mut x| {
                    x.assign(&vec_xs);
                }
            );

            let rslt = rslt?;
            vec_xs.assign(&rslt.rows(0 .. n + 1));

            t *= self.mu;
        }

        Ok(vec_xs.rows(0 .. n).clone_sz())
    }
}
