//! Quadratically constrained quadratic program

use super::prelude::*;

use std::io::Write;

/// Quadratically constrained quadratic program
/// 
/// <script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML' async></script>
/// 
/// The problem is
/// \\[
/// \\begin{array}{ll}
/// {\\rm minimize} & {1 \\over 2} x^T P_0 x + q_0^T x + r_0 \\\\
/// {\\rm subject \\ to} & {1 \\over 2} x^T P_i x + q_i^T x + r_i \\le 0 \\quad (i = 1, \\ldots, m) \\\\
/// & A x = b,
/// \\end{array}
/// \\]
/// where
/// - variables \\( x \\in {\\bf R}^n \\)
/// - \\( P_j \\in {\\bf S}_{+}^n \\), \\( q_j \\in {\\bf R}^n \\), \\( r_j \\in {\\bf R} \\) for \\( j = 0, \\ldots, m \\)
/// - \\( A \\in {\\bf R}^{p \\times n} \\), \\( b \\in {\\bf R}^p \\).
/// 
/// Internally a slack variable \\( s \\in {\\bf R} \\) is introduced for the infeasible start method as follows:
/// \\[
/// \\begin{array}{ll}
/// {\\rm minimize}_{x,s} & {1 \\over 2} x^T P_0 x + q_0^T x + r_0 \\\\
/// {\\rm subject \\ to} & {1 \\over 2} x^T P_i x + q_i^T x + r_i \\le s \\quad (i = 1, \\ldots, m) \\\\
/// & A x = b \\\\
/// & s = 0.
/// \\end{array}
/// \\]
pub trait QCQP {
    fn solve_qcqp<L>(&mut self, param: &PDIPMParam, log: &mut L,
                     mat_p: &[Mat], vec_q: &[Mat], scl_r: &[FP],
                     mat_a: &Mat, vec_b: &Mat)
                     -> Result<Mat, String>
    where L: Write;
}

fn check_param(mat_p: &[Mat], vec_q: &[Mat], scl_r: &[FP],
               mat_a: &Mat, vec_b: &Mat)
               -> Result<(usize, usize, usize), String>
{
        if mat_p.len() == 0 {return Err("mat_p: 0 length".into());}

        let (n, _) = mat_p[0].size();
        let m = mat_p.len() - 1;
        let (p, _) = mat_a.size();

        if n == 0 {return Err("mat_p[0]: 0 rows".into());}
        // m = 0 means NO inequality constraints
        // p = 0 means NO equality constraints

        if vec_q.len() != m + 1 {return Err(format!("vec_q: length {} must be {}", vec_q.len(), m + 1));}
        if scl_r.len() != m + 1 {return Err(format!("scl_r: length {} must be {}", scl_r.len(), m + 1));}

        for i in 0 ..= m {
            if mat_p[i].size() != (n, n) {return Err(format!("mat_p[{}]: size {:?} must be {:?}", i, mat_p[i].size(), (n, n)));}
            if vec_q[i].size() != (n, 1) {return Err(format!("vec_q[{}]: size {:?} must be {:?}", i, vec_q[i].size(), (n, 1)));}
        }

        if mat_a.size() != (p, n) {return Err(format!("mat_a: size {:?} must be {:?}", mat_a.size(), (p, n)));}
        if vec_b.size() != (p, 1) {return Err(format!("vec_b: size {:?} must be {:?}", vec_b.size(), (p, 1)));}

        Ok((n, m, p))
}

impl QCQP for PDIPM
{
    /// Runs the solver with given parameters.
    /// 
    /// Returns `Ok` with optimal \\(x\\) or `Err` with message string.
    /// * `param` is solver parameters.
    /// * `log` outputs solver progress.
    /// * `mat_p` is \\(P_0, \\ldots, P_m\\).
    /// * `vec_q` is \\(q_0, \\ldots, q_m\\).
    /// * `scl_r` is \\(r_0, \\ldots, r_m\\).
    /// * `mat_a` is \\(A\\).
    /// * `vec_b` is \\(b\\).
    fn solve_qcqp<L>(&mut self, param: &PDIPMParam, log: &mut L,
                     mat_p: &[Mat], vec_q: &[Mat], scl_r: &[FP],
                     mat_a: &Mat, vec_b: &Mat)
                     -> Result<Mat, String>
    where L: Write
    {
        // ----- parameter check

        let (n, m, p) = check_param(mat_p, vec_q, scl_r, mat_a, vec_b)?;

        // ----- initial value of a slack variable

        let s: FP = *scl_r.iter().max_by(|l, r| l.partial_cmp(r).unwrap()).unwrap();
        let mut margin = param.margin;
        let mut s_initial = s + margin;
        while s_initial <= s {
            margin *= 2.;
            s_initial = s + margin;
        }

        // ----- start to solve

        let rslt = self.solve(param, log,
            n + 1, m, p + 1, // '+ 1' is for a slack variable
            |x, df_o| {
                df_o.rows_mut(0 .. n).assign(&(
                    &mat_p[0] * x.rows(0 .. n) + &vec_q[0]
                ));
                // for a slack variable
                df_o[(n, 0)] = 0.;
            },
            |_, ddf_o| {
                ddf_o.slice_mut(0 .. n, 0 .. n).assign(&mat_p[0]);
                // for a slack variable
                ddf_o.row_mut(n).assign_all(0.);
                ddf_o.col_mut(n).assign_all(0.);
            },
            |x, f_i| {
                for r in 0 .. m {
                    let i = r + 1;
                    let xn = x.rows(0 .. n);
                    let tmp = xn.prod(&(&mat_p[i] * &xn)) / 2.
                            + vec_q[i].prod(&xn)
                            + scl_r[i];
                    f_i[(r, 0)] = tmp - x[(n, 0)]; // minus a slack variable
                }
            },
            |x, df_i| {
                for r in 0 .. m {
                    let i = r + 1;
                    let xn = x.rows(0 .. n);
                    let tmp = &mat_p[i] * xn + &vec_q[i];
                    df_i.slice_mut(r ..= r, 0 .. n).assign(&tmp.t());
                    // for a slack variable
                    df_i[(r, n)] = -1.;
                }
            },
            |_, ddf_i, i| {
                ddf_i.slice_mut(0 .. n, 0 .. n).assign(&mat_p[i + 1]);
                // for a slack variable
                ddf_i.row_mut(n).assign_all(0.);
                ddf_i.col_mut(n).assign_all(0.);
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
            Err(s) => Err(s.into())
        }
    }
}
