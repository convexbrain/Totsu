use super::mat::{Mat, FP};
use super::pdipm::PDIPM;

use std::io::Write;

pub trait QP {
    fn solve_qp<L>(&self, log: L,
                   mat_p: &Mat, vec_q: &Mat,
                   mat_g: &Mat, vec_h: &Mat,
                   mat_a: &Mat, vec_b: &Mat)
                   -> Result<Mat, &'static str>
    where L: Write;
}

fn check_param(mat_p: &Mat, vec_q: &Mat,
               mat_g: &Mat, vec_h: &Mat,
               mat_a: &Mat, vec_b: &Mat)
               -> Result<(usize, usize, usize), &'static str>
{
    let (n, _) = mat_p.size();
    let (m, _) = mat_g.size();
    let (p, _) = mat_a.size();

    if n == 0 {return Err("mat_p: 0 rows");}
    // m = 0 means NO inequality constraints
    // p = 0 means NO equality constraints

    if mat_p.size() != (n, n) {return Err("mat_p: size mismatch");}
    if vec_q.size() != (n, 1) {return Err("vec_q: size mismatch");}
    if mat_g.size() != (m, n) {return Err("mat_g: size mismatch");}
    if vec_h.size() != (m, 1) {return Err("vec_h: size mismatch");}
    if mat_a.size() != (p, n) {return Err("mat_a: size mismatch");}
    if vec_b.size() != (p, 1) {return Err("vec_b: size mismatch");}

    Ok((n, m, p))
}

impl QP for PDIPM
{
    fn solve_qp<L>(&self, log: L,
                   mat_p: &Mat, vec_q: &Mat,
                   mat_g: &Mat, vec_h: &Mat,
                   mat_a: &Mat, vec_b: &Mat)
                   -> Result<Mat, &'static str>
    where L: Write
    {
        // ----- parameter check

        let (n, m, p) = check_param(mat_p, vec_q, mat_g, vec_h, mat_a, vec_b)?;

        // ----- initial value of a slack variable

        let s = -vec_h.min().2;
        let mut margin = self.margin;
        let mut s_inital = s + margin;
        while s_inital <= s {
            margin *= 2.;
            s_inital = s + margin;
        }

        // ----- start to solve

        let rslt = self.solve(n + 1, m, p + 1, // '+ 1' is for a slack variable
            log,
            |x, df_o| {
                df_o.rows_mut(0 .. n).assign(&(
                    mat_p * x.rows(0 .. n) + vec_q
                ));
                // for a slack variable
                df_o[(n, 0)] = 0.;
            },
            |_, ddf_o| {
                ddf_o.slice_mut(0 .. n, 0 .. n).assign(mat_p);
                // for a slack variable
                ddf_o.row_mut(n).assign_all(0.);
                ddf_o.col_mut(n).assign_all(0.);
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
            Ok(y) => Ok(y.rows(0 .. n).clone_sz()),
            Err(s) => Err(s)
        }
    }
}
