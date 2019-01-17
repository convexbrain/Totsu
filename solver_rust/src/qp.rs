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
        let mut margin = self.margin;
        let mut s_inital = s + margin;
        while s_inital <= s {
            s_inital = s + margin;
            margin *= 2.;
        }

        // ----- start to solve

        let rslt = self.solve(n + 1, m, p + 1, // '+ 1' is for a slack variable
            log,
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
}
