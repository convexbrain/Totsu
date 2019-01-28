//! Linear program

use super::prelude::*;

use std::io::Write;

/// Linear program
/// 
/// <script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML' async></script>
/// 
/// The problem is
/// \\[
/// \\begin{array}{ll}
/// {\\rm minimize} & c^Tx + d \\\\
/// {\\rm subject \\ to} & G x \\preceq h \\\\
/// & A x = b,
/// \\end{array}
/// \\]
/// where
/// - variables \\( x \\in {\\bf R}^n \\)
/// - \\( c \\in {\\bf R}^n \\), \\( d \\in {\\bf R} \\)
/// - \\( G \\in {\\bf R}^{m \\times n} \\), \\( h \\in {\\bf R}^m \\)
/// - \\( A \\in {\\bf R}^{p \\times n} \\), \\( b \\in {\\bf R}^p \\).
/// 
/// Internally a slack variable \\( s \\in {\\bf R} \\) is introduced for the infeasible start method as follows:
/// \\[
/// \\begin{array}{ll}
/// {\\rm minimize}_{x,s} & c^T x + d \\\\
/// {\\rm subject \\ to} & G x \\preceq h + s {\\bf 1} \\\\
/// & A x = b \\\\
/// & s = 0.
/// \\end{array}
/// \\]
/// 
/// In the following, \\( d \\) does not appear since it does not matter.
pub trait LP {
    fn solve_lp<L>(&mut self, param: &PDIPMParam, log: &mut L,
                   vec_c: &Mat,
                   mat_g: &Mat, vec_h: &Mat,
                   mat_a: &Mat, vec_b: &Mat)
                   -> Result<Mat, &'static str>
    where L: Write;
}

// TODO: error string
fn check_param(vec_c: &Mat,
               mat_g: &Mat, vec_h: &Mat,
               mat_a: &Mat, vec_b: &Mat)
               -> Result<(usize, usize, usize), &'static str>
{
    let (n, _) = vec_c.size();
    let (m, _) = mat_g.size();
    let (p, _) = mat_a.size();

    if n == 0 {return Err("vec_c: 0 rows");}
    // m = 0 means NO inequality constraints
    // p = 0 means NO equality constraints

    if vec_c.size() != (n, 1) {return Err("vec_c: size mismatch");}
    if mat_g.size() != (m, n) {return Err("mat_g: size mismatch");}
    if vec_h.size() != (m, 1) {return Err("vec_h: size mismatch");}
    if mat_a.size() != (p, n) {return Err("mat_a: size mismatch");}
    if vec_b.size() != (p, 1) {return Err("vec_b: size mismatch");}

    Ok((n, m, p))
}

impl LP for PDIPM
{
    /// Runs the solver with given parameters.
    /// 
    /// Returns `Ok` with optimal \\(x\\) or `Err` with message string.
    /// * `param` is solver parameters.
    /// * `log` outputs solver progress.
    /// * `vec_c` is \\(c\\).
    /// * `mat_g` is \\(G\\).
    /// * `vec_h` is \\(h\\).
    /// * `mat_a` is \\(A\\).
    /// * `vec_b` is \\(b\\).
    fn solve_lp<L>(&mut self, param: &PDIPMParam, log: &mut L,
                   vec_c: &Mat,
                   mat_g: &Mat, vec_h: &Mat,
                   mat_a: &Mat, vec_b: &Mat)
                   -> Result<Mat, &'static str>
    where L: Write
    {
        // ----- parameter check

        let (n, m, p) = check_param(vec_c, mat_g, vec_h, mat_a, vec_b)?;

        // ----- initial value of a slack variable

        let s = -vec_h.min().unwrap_or(0.);
        let mut margin = param.margin;
        let mut s_initial = s + margin;
        while s_initial <= s {
            margin *= 2.;
            s_initial = s + margin;
        }

        // ----- start to solve

        let rslt = self.solve(param, log,
            n + 1, m, p + 1, // '+ 1' is for a slack variable
            |_, df_o| {
                df_o.rows_mut(0 .. n).assign(&vec_c);
                // for a slack variable
                df_o[(n, 0)] = 0.;
            },
            |_, ddf_o| {
                ddf_o.assign_all(0.);
            },
            |x, f_i| {
                f_i.assign(&(
                    mat_g * x.rows(0 .. n) - vec_h
                    - x[(n, 0)] * (m as FP) // minus a slack variable
                ))
            },
            |_, df_i| {
                df_i.cols_mut(0 .. n).assign(&mat_g);
                // for a slack variable
                df_i.col_mut(n).assign_all(-(m as FP));
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
                x.assign_all(0.);
                x[(n, 0)] = s_initial;
            }
        );

        match rslt {
            Ok(y) => Ok(y.rows(0 .. n).clone_sz()),
            Err(s) => Err(s)
        }
    }
}
