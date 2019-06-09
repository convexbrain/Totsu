//! Matrix singular value decomposition

use super::mat::{Mat, FP, FP_EPSILON};
use super::spmat::SpMat;


const TOL_CNV1_SQ: FP = FP_EPSILON * FP_EPSILON * 4.;
const TOL_CNV2_SQ: FP = FP_EPSILON * FP_EPSILON;
const TOL_SINV_SQ: FP = FP_EPSILON * FP_EPSILON;

/// Matrix singular value decomposition
#[derive(Debug)]
pub struct SVDS
{
    u: Mat,
    vt_h: Mat
}

// TODO: refactor along with matsvd
impl SVDS
{
    pub fn new(sz: (usize, usize)) -> SVDS
    {
        SVDS {
            u: Mat::new(sz.1, sz.0),
            vt_h: Mat::new_vec(0)
        }
    }

    pub fn spsolve(&mut self, mat_a: &SpMat, mat_b: &Mat) -> Mat
    {
        assert_eq!(mat_a.size().0, mat_b.size().0);

        self.u.assign(&mat_a.t());
        self.vt_h = mat_b.clone_sz();

        self.decomp();

        self.solve()
    }

    fn apply_jacobi_rot(&mut self, c1: usize, c2: usize) -> bool
    {
        let a = self.u.col(c1).norm_p2sq();
        let b = self.u.col(c2).norm_p2sq();
        let d = self.u.col(c1).prod(&self.u.col(c2));

        if (d * d <= TOL_CNV1_SQ * a * b) || (d * d <= TOL_CNV2_SQ) {
            true
        }
        else {
            let zeta = (b - a) / (2.0 * d);
            let t = if zeta > 0.0 {
                1.0 / (zeta + FP::sqrt(1.0 + zeta * zeta))
            }
            else {
                -1.0 / (-zeta + FP::sqrt(1.0 + zeta * zeta))
            };
            let c = 1.0 / FP::sqrt(1.0 + t * t);
            let s = c * t;

            let tmp1 = self.u.col(c1) * c - self.u.col(c2) * s;
            let tmp2 = self.u.col(c1) * s + self.u.col(c2) * c;
            self.u.col_mut(c1).assign(&tmp1);
            self.u.col_mut(c2).assign(&tmp2);

            let tmp1 = self.vt_h.row(c1) * c - self.vt_h.row(c2) * s;
            let tmp2 = self.vt_h.row(c1) * s + self.vt_h.row(c2) * c;
            self.vt_h.row_mut(c1).assign(&tmp1);
            self.vt_h.row_mut(c2).assign(&tmp2);

            false
        }
    }
    //
    fn decomp(&mut self)
    {
        let (_, n) = self.u.size();

        let mut converged_all = false;
        while !converged_all {
            converged_all = true;

            for i in 0 .. n - 1 {
                for j in i + 1 .. n {
                    if !self.apply_jacobi_rot(i, j) {converged_all = false;}
                }
            }
        }
    }
    //
    fn solve(&mut self) -> Mat
    {
        let (_, n) = self.u.size();

        for i in 0 .. n {
            let ut_col = self.u.col(i);
            let mut vt_h_row = self.vt_h.row_mut(i);

            let sigma_sq = ut_col.norm_p2sq();

            if sigma_sq < TOL_SINV_SQ {
                vt_h_row.assign_all(0.);
            }
            else {
                vt_h_row /= sigma_sq;
            }
        }

        &self.u * &self.vt_h
    }
}

// TODO: rename
pub fn solve(g: &Mat, h: &Mat) -> Mat
{
    let mut s = SVDS::new(g.size());

    assert_eq!(g.size().0, h.size().0);

    s.u.assign(&g.t());
    s.vt_h = h.clone_sz();

    s.decomp();

    s.solve()
}
