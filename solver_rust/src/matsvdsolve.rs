//! Matrix singular value decomposition

use super::mat::{Mat, FP, FP_EPSILON};
use super::spmat::SpMat;


const TOL_CNV1_SQ: FP = FP_EPSILON * FP_EPSILON * 4.;
const TOL_CNV2_SQ: FP = FP_EPSILON * FP_EPSILON;
const TOL_SINV_SQ: FP = FP_EPSILON * FP_EPSILON;


struct OneSidedJacobi
{
    u: Mat,
    v: Mat
}

impl OneSidedJacobi
{
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

            let tmp1 = self.v.col(c1) * c - self.v.col(c2) * s;
            let tmp2 = self.v.col(c1) * s + self.v.col(c2) * c;
            self.v.col_mut(c1).assign(&tmp1);
            self.v.col_mut(c2).assign(&tmp2);

            false
        }
    }
    //
    fn decomp(&mut self)
    {
        let (_, n) = self.u.size();

        assert_eq!(n, self.v.size().1);

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
}


/// Matrix singular value decomposition
pub struct SVDS
{
    j: OneSidedJacobi
}

// TODO: refactor along with matsvd
impl SVDS
{
    pub fn new(sz: (usize, usize)) -> SVDS
    {
        SVDS {
            j: OneSidedJacobi {
                u: Mat::new(sz.1, sz.0),
                v: Mat::new_vec(0)
            }
        }
    }

    pub fn spsolve(&mut self, g: &SpMat, h: &Mat) -> Mat
    {
        assert_eq!(g.size().0, h.size().0);

        self.j.u.assign(&g.t());
        self.j.v = h.t().clone_sz();

        self.do_solve()
    }

    pub fn solve(&mut self, g: &Mat, h: &Mat) -> Mat
    {
        assert_eq!(g.size().0, h.size().0);

        self.j.u.assign(&g.t());
        self.j.v = h.t().clone_sz();

        self.do_solve()
    }

    fn do_solve(&mut self) -> Mat
    {
        self.j.decomp();

        let (_, n) = self.j.u.size();

        for i in 0 .. n {
            let u_col = self.j.u.col(i);
            let mut v_col = self.j.v.col_mut(i);

            let sigma_sq = u_col.norm_p2sq();

            if sigma_sq < TOL_SINV_SQ {
                v_col.assign_all(0.);
            }
            else {
                v_col /= sigma_sq;
            }
        }

        &self.j.u * &self.j.v.t()
    }
}
