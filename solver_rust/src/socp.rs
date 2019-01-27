//! Second-order cone program

use super::prelude::*;

use std::io::Write;

/// Second-order cone program
/// 
/// <script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML' async></script>
/// 
/// The problem is
/// \\[
/// \\begin{array}{ll}
/// {\\rm minimize} & f^T x \\\\
/// {\\rm subject \\ to} & \\| G_i x + h_i \\|_2 \\le c_i^T x + d_i \\quad (i = 0, \\ldots, m - 1) \\\\
/// & A x = b,
/// \\end{array}
/// \\]
/// where
/// - variables \\( x \\in {\\bf R}^n \\)
/// - \\( f \\in {\\bf R}^n \\)
/// - \\( G_i \\in {\\bf R}^{n_i \\times n} \\), \\( h_i \\in {\\bf R}^{n_i} \\), \\( c_i \\in {\\bf R}^n \\), \\( d_i \\in {\\bf R} \\)
/// - \\( A \\in {\\bf R}^{p \\times n} \\), \\( b \\in {\\bf R}^p \\).
/// 
/// Internally an **approximately equivalent** problem is formed and
/// an auxiliary variable \\( s \\in {\\bf R}^m \\) is introduced for the infeasible start method as follows:
/// \\[
/// \\begin{array}{lll}
/// {\\rm minimize}\_{x,s} & f^T x \\\\
/// {\\rm subject \\ to} & {\\| G_i x + h_i \\|\_2^2 \\over s_i} \\le s_i & (i = 0, \\ldots, m - 1) \\\\
/// & s_i \\ge \\epsilon_{\\rm bd} & (i = 0, \\ldots, m - 1) \\\\
/// & c_i^T x + d_i = s_i & (i = 0, \\ldots, m - 1) \\\\
/// & A x = b,
/// \\end{array}
/// \\]
/// where \\( \\epsilon_{\\rm bd} > 0 \\) indicates the extent of approximation that excludes \\( c_i^T x + d_i = 0 \\) boundary.
pub trait SOCP {
    fn solve_socp<L>(&self, log: &mut L,
                     vec_f: &Mat,
                     mat_g: &[Mat], vec_h: &[Mat], vec_c: &[Mat], scl_d: &[FP],
                     mat_a: &Mat, vec_b: &Mat)
                     -> Result<Mat, &'static str>
    where L: Write;
}

fn check_param(vec_f: &Mat,
               mat_g: &[Mat], vec_h: &[Mat], vec_c: &[Mat], scl_d: &[FP],
               mat_a: &Mat, vec_b: &Mat)
               -> Result<(usize, usize, usize), &'static str>
{
        let (n, _) = vec_f.size();
        let m = mat_g.len();
        let (p, _) = mat_a.size();

        if n == 0 {return Err("vec_f: 0 rows");}
        // m = 0 means NO inequality constraints
        // p = 0 means NO equality constraints

        if vec_h.len() != m {return Err("vec_h: length mismatch");}
        if vec_c.len() != m {return Err("vec_c: length mismatch");}
        if scl_d.len() != m {return Err("scl_d: length mismatch");}

        if vec_f.size() != (n, 1) {return Err("vec_c: size mismatch");}

        for i in 0 .. m {
            let (ni, _) = mat_g[i].size();
            if mat_g[i].size() != (ni, n) {return Err("mat_g[_]: size mismatch");}
            if vec_h[i].size() != (ni, 1) {return Err("vec_h[_]: size mismatch");}
            if vec_c[i].size() != (n, 1) {return Err("vec_c[_]: size mismatch");}
        }

        if mat_a.size() != (p, n) {return Err("mat_a: size mismatch");}
        if vec_b.size() != (p, 1) {return Err("vec_b: size mismatch");}

        Ok((n, m, p))
}

impl SOCP for PDIPM
{
    /// Runs the solver with given parameters.
    /// 
    /// Returns `Ok` with optimal \\(x\\) or `Err` with message string.
    /// * `log` outputs solver progress.
    /// * `vec_f` is \\(f\\).
    /// * `mat_g` is \\(G_0, \\ldots, G_{m-1}\\).
    /// * `vec_h` is \\(h_0, \\ldots, h_{m-1}\\).
    /// * `vec_c` is \\(c_0, \\ldots, c_{m-1}\\).
    /// * `scl_d` is \\(d_0, \\ldots, d_{m-1}\\).
    /// * `mat_a` is \\(A\\).
    /// * `vec_b` is \\(b\\).
    fn solve_socp<L>(&self, log: &mut L,
                     vec_f: &Mat,
                     mat_g: &[Mat], vec_h: &[Mat], vec_c: &[Mat], scl_d: &[FP],
                     mat_a: &Mat, vec_b: &Mat)
                     -> Result<Mat, &'static str>
    where L: Write
    {
        // ----- parameter check

        let (n, m, p) = check_param(vec_f, mat_g, vec_h, vec_c, scl_d, mat_a, vec_b)?;

        let eps_div0 = self.eps;
        let eps_bd = self.eps;

        // ----- start to solve

        let rslt = self.solve(n + m, m + m, p + m, // '+ m' is for slack variables
            log,
            |_, df_o| {
                df_o.rows_mut(0 .. n).assign(&vec_f);
                // for slack variables
                df_o.rows_mut(n .. n + m).assign_all(0.);
            },
            |_, ddf_o| {
                ddf_o.assign_all(0.);
            },
            |x, f_i| {
                let xn = x.rows(0 .. n);

                for r in 0 .. m {
                    let xnr = x[(n + r, 0)];

                    let inv_s = if xnr.abs() > eps_div0 {
                        1. / xnr
                    }
                    else {
                        // guard from div by zero
                        1. / eps_div0
                    };

                    let tmp = &mat_g[r] * &xn + &vec_h[r];
                    f_i[(r, 0)] = tmp.norm_p2sq() * inv_s - xnr;

                    // for slack variables
                    f_i[(r + m, 0)] = eps_bd - xnr;
                }
            },
            |x, df_i| {
                let xn = x.rows(0 .. n);

                df_i.assign_all(0.);

                for r in 0 .. m {
                    let xnr = x[(n + r, 0)];

                    let inv_s = if xnr.abs() > eps_div0 {
                        1. / xnr
                    }
                    else {
                        // guard from div by zero
                        1. / eps_div0
                    };

                    let tmp1 = &mat_g[r] * &xn + &vec_h[r];
                    let tmp1_norm_p2sq = tmp1.norm_p2sq();
                    let tmp2 = 2. * inv_s * mat_g[r].t() * tmp1;
                    df_i.slice_mut(r ..= r, 0 .. n).assign(&tmp2.t());

                    // for slack variables
                    df_i[(r, n + r)] = -inv_s * inv_s * tmp1_norm_p2sq - 1.;

                    // for slack variables
                    df_i[(r + m, n + r)] = -1.;
                }
            },
            |x, ddf_i, i| {
                ddf_i.assign_all(0.); // for slack variables

                if i < m {
                    let xn = x.rows(0 .. n);
                    let xni = x[(n + i, 0)];

                    let inv_s =  if xni.abs() > eps_div0 {
                        1. / xni
                    }
                    else {
                        // guard from div by zero
                        1. / eps_div0
                    };

                    ddf_i.slice_mut(0 .. n, 0 .. n).assign(&(
                        2. * inv_s * mat_g[i].t() * &mat_g[i]
                    ));

                    let tmp1 = &mat_g[i] * xn + &vec_h[i];
                    let tmp1_norm_p2sq = tmp1.norm_p2sq();
                    let tmp2 = -2. * inv_s * inv_s * mat_g[i].t() * tmp1;

                    // for slack variables
                    ddf_i.slice_mut(0 .. n, n + i ..= n + i).assign(&tmp2);

                    // for slack variables
                    ddf_i.slice_mut(n + i ..= n + i, 0 .. n).assign(&tmp2.t());

                    // for slack variables
                    ddf_i[(n + i, n + i)] = 2. * inv_s * inv_s * inv_s * tmp1_norm_p2sq;
                }
            },
            |a, b| {
                a.slice_mut(0 .. p, 0 .. n).assign(mat_a);
                b.rows_mut(0 .. p).assign(vec_b);

                // for a slack variable
                for r in 0 .. m {
                    a.slice_mut(p + r ..= p + r, 0 .. n).assign(&vec_c[r].t());
                    a[(p + r, n + r)] = -1.;
                    b[(p + r, 0)] = -scl_d[r];
                }
            },
            |mut x| {
                // slack variables
                for i in 0 .. m {
                    let s = vec_h[i].norm_p2() + eps_bd;

                    let mut margin = self.margin;
                    let mut s_initial = s + margin;
                    while s_initial <= s {
                        margin *= 2.;
                        s_initial = s + margin;
                    }
                    x[(n + i, 0)] = s_initial;
                }
            },
            || true
        );

        match rslt {
            Ok(y) => Ok(y.rows(0 .. n).clone_sz()),
            Err(s) => Err(s)
        }
    }
}
