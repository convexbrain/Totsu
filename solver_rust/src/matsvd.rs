/*!
Matrix singular value decomposition

References
* [LAPACK Working Notes](https://www.netlib.org/lapack/lawns/index.html) lawn15
* J. Demmel and K. Veselic, "Jacobi’s Method is More Accurate than QR,"
  UT-CS-89-88, October 1989.
*/

use super::mat::{Mat, FP, FP_EPSILON, FP_MINPOS};
use super::spmat::SpMat;


const TOL_CNV1_SQ: FP = FP_EPSILON * FP_EPSILON * 4.;
const TOL_CNV2_SQ: FP = FP_EPSILON * FP_EPSILON;
const TOL_SINV: FP = FP_EPSILON;
const TOL_SINV_SQ: FP = TOL_SINV * TOL_SINV;
const TOL_DIV0: FP = FP_MINPOS;


#[derive(Debug)]
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


/// Linear equation solver by SVD
#[derive(Debug)]
pub struct SVDS
{
    j: OneSidedJacobi
}

impl SVDS
{
    /// Makes a SVD workplace for factorizing a specified size matrix.
    pub fn new(sz: (usize, usize)) -> SVDS
    {
        SVDS {
            j: OneSidedJacobi {
                u: Mat::new(sz.1, sz.0),
                v: Mat::new_vec(0)
            }
        }
    }
    //
    /// Solves sparse matrix linear equations.
    pub fn spsolve(&mut self, g: &SpMat, h: &Mat) -> Mat
    {
        assert_eq!(g.size().0, h.size().0);

        self.j.u.assign(&g.t());
        self.j.v = h.t().clone_sz();

        self.do_solve()
    }
    //
    /// Solves linear equations.
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


/// Matrix singular value decomposition
#[derive(Debug)]
pub struct MatSVD
{
    transposed: bool,
    //
    j: OneSidedJacobi,
    s: Mat
}

impl MatSVD
{
    /// Makes a SVD workplace for factorizing a specified size matrix.
    pub fn new((nrows, ncols): (usize, usize)) -> MatSVD
    {
        let transposed = nrows < ncols;

        let (u_nrows, u_ncols) = if !transposed {
            (nrows, ncols)
        }
        else {
            (ncols, nrows)
        };

        let svd = MatSVD {
            transposed,
            j: OneSidedJacobi {
                u: Mat::new(u_nrows, u_ncols),
                v: Mat::new(u_ncols, u_ncols).set_eye(1.)
            },
            s: Mat::new_vec(u_ncols),
        };

        svd
    }
    //
    fn norm_singular(&mut self)
    {
        let (_, n) = self.j.u.size();

        for i in 0 .. n {
            let mut col = self.j.u.col_mut(i);
            let s = col.norm_p2();
            self.s[(i, 0)] = s;

            if s.abs() < TOL_DIV0 {
                continue;
            }

            col /= s;
        }
    }
    //
    /// Runs SVD of a specified matrix.
    pub fn decomp(&mut self, g: &Mat)
    {
        if !self.transposed {
            self.j.u.assign(g);
        }
        else {
            self.j.u.assign(&g.t());
        }

        self.j.v.assign_eye(1.);

        self.j.decomp();

        self.norm_singular();
    }
    //
    /// Runs SVD of a specified matrix with a warm-start from the last SVD result.
    pub fn decomp_warm(&mut self, g: &Mat)
    {
        if !self.transposed {
            self.j.u.assign(&(g * &self.j.v));
        }
        else {
            self.j.u.assign(&(g.t() * &self.j.v));
        }

        self.j.decomp();

        self.norm_singular();
    }
    //
    /// Solves linear equations using the last SVD result.
    pub fn solve(&self, h: &Mat) -> Mat
    {
        let sinv = Mat::new_like(&self.s).set_by(|r, _| {
            let s = self.s[(r, 0)];

            if s.abs() < TOL_SINV {
                0.
            }
            else {
                1. / s
            }
        });

        if !self.transposed {
            &self.j.v * sinv.diag_mul(&(self.j.u.t() * h))
        }
        else {
            &self.j.u * sinv.diag_mul(&(self.j.v.t() * h))
        }
    }
    //
    /// Returns singular values.
    pub fn s(&self) -> &Mat
    {
        &self.s
    }
}

#[cfg(test)]
use super::mat::XOR64;

#[test]
fn test_decomp()
{
    const TOL_RMSE: FP = 1.0 / (1u64 << 32) as FP;

    let mut r = XOR64::init();
    let mat = Mat::new(4, 4).set_by(|_, _| r.next());
    println!("mat = {}", mat);

    let mut svd = MatSVD::new(mat.size());

    svd.decomp(&mat);

    //

    let g = if !svd.transposed {
        &svd.j.u * svd.s.diag_mul(&svd.j.v.t())
    }
    else {
        &svd.j.v * svd.s.diag_mul(&svd.j.u.t())
    };
    println!("mat reconstructed = {}", g);

    let g_size = g.size();
    let g_err = (g - mat).norm_p2sq() / ((g_size.0 * g_size.1) as FP);
    println!("g_err = {:e}", g_err);
    assert!(g_err < TOL_RMSE);

    //

    let mut utu = svd.j.u.t() * &svd.j.u;
    println!("u' * u = {}", utu);

    let utu_size = utu.size();
    utu.assign_by(|r, c| if r == c {Some(0.)} else {None});
    let utu_err = utu.norm_p2sq() / ((utu_size.0 * utu_size.1) as FP);
    println!("utu_err = {:e}", utu_err);
    assert!(utu_err < TOL_RMSE);

    //

    let mut vvt = &svd.j.v * svd.j.v.t();
    println!("v * v' = {}", vvt);

    let vvt_size = vvt.size();
    vvt.assign_by(|r, c| if r == c {Some(0.)} else {None});
    let vvt_err = vvt.norm_p2sq() / ((vvt_size.0 * vvt_size.1) as FP);
    println!("vvt_err = {:e}", vvt_err);
    assert!(vvt_err < TOL_RMSE);
}

#[test]
fn test_solve()
{
    const TOL_RMSE: FP = 1.0 / (1u64 << 32) as FP;

    let mat = Mat::new(2, 2).set_iter(&[
        1., 2.,
        3., 4.
    ]);

    let mut svd = MatSVD::new(mat.size());

    svd.decomp(&mat);

    let vec = Mat::new_vec(2).set_iter(&[
        5., 6.
    ]);

    let x = svd.solve(&vec);
    println!("x = {}", x);

    let h = &mat * &x;
    println!("vec reconstructed = {}", h);

    let h_size = h.size();
    let h_err = (h - vec).norm_p2sq() / ((h_size.0 * h_size.1) as FP);
    assert!(h_err < TOL_RMSE);
}
