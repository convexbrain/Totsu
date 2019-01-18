use super::mat::{Mat, FP};
use super::pdipm::PDIPM;

use std::io::Write;

pub trait QCQP {
    fn solve_qcqp<L>(&self, log: L,
                     mat_p: &[Mat], vec_q: &[Mat], scl_r: &[FP],
                     mat_a: &Mat, vec_b: &Mat)
                     -> Result<Mat, &'static str>
    where L: Write;
}

fn check_param(mat_p: &[Mat], vec_q: &[Mat], scl_r: &[FP],
               mat_a: &Mat, vec_b: &Mat)
               -> Result<(usize, usize, usize), &'static str>
{
        if mat_p.len() == 0 {return Err("mat_p: 0 length");}

        let (n, _) = mat_p[0].size();
        let m = mat_p.len() - 1;
        let (p, _) = mat_a.size();

        if n == 0 {return Err("mat_p[_]: 0 rows");}
        // m = 0 means NO inequality constraints
        // p = 0 means NO equality constraints

        for i in 0 ..= m {
            if mat_p[i].size() != (n, n) {return Err("mat_p[_]: size mismatch");}
        }

        if vec_q.len() != m + 1 {return Err("vec_q: length mismatch");}

        for i in 0 ..= m {
            if vec_q[i].size() != (n, 1) {return Err("vec_q[_]: size mismatch");}
        }

        if scl_r.len() != m + 1 {return Err("scl_r: length mismatch");}

        if mat_a.size() != (p, n) {return Err("mat_a: size mismatch");}
        if vec_b.size() != (p, 1) {return Err("vec_b: size mismatch");}

        Ok((n, m, p))
}

impl QCQP for PDIPM
{
    fn solve_qcqp<L>(&self, log: L,
                     mat_p: &[Mat], vec_q: &[Mat], scl_r: &[FP],
                     mat_a: &Mat, vec_b: &Mat)
                     -> Result<Mat, &'static str>
    where L: Write
    {
        // ----- parameter check

        let (n, m, p) = check_param(mat_p, vec_q, scl_r, mat_a, vec_b)?;

        // ----- initial value of a slack variable

        let s: FP = *scl_r.iter().max_by(|l, r| l.partial_cmp(r).unwrap()).unwrap();
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
                    let tmp = xn.t() * &mat_p[i] * &xn / 2.
                            + vec_q[i].t() * &xn
                            + scl_r[i];
                    f_i[(r, 0)] = tmp[(0, 0)] - x[(n, 0)]; // minus a slack variable
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
