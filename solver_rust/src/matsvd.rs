//! Matrix singular value decomposition

use super::mat::{Mat, FP, FP_EPSILON, FP_MINPOS};

const TOL_CNV2: FP = FP_EPSILON * FP_EPSILON * 4.;
const TOL_DIV0: FP = FP_MINPOS;
const TOL_SINV: FP = FP_EPSILON;

/// Matrix singular value decomposition
#[derive(Debug)]
pub struct MatSVD
{
    transposed: bool,
    //
    u: Mat,
    s: Mat,
    v: Mat
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
            u: Mat::new(u_nrows, u_ncols),
            s: Mat::new_vec(u_ncols),
            v: Mat::new(u_ncols, u_ncols).set_eye(1.)
        };

        svd
    }
    //
    fn apply_jacobi_rot(&mut self, c1: usize, c2: usize) -> bool
    {
        let a = self.u.col(c1).norm_p2sq();
        let b = self.u.col(c2).norm_p2sq();
        let d = self.u.col(c1).prod(&self.u.col(c2));

        if (d * d <= TOL_CNV2 * a * b) || (d.abs() <= TOL_CNV2) {
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
    fn norm_singular(&mut self)
    {
        let (_, n) = self.u.size();

        for i in 0 .. n {
            let mut col = self.u.col_mut(i);
            let s = col.norm_p2();
            self.s[(i, 0)] = s;

            if s.abs() < TOL_DIV0 {
                continue;
            }

            col /= s;
        }
    }
    //
    fn do_decomp(&mut self)
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

        self.norm_singular();
    }
    //
    /// Runs SVD of a specified matrix.
    pub fn decomp(&mut self, g: &Mat)
    {
        if !self.transposed {
            self.u.assign(g);
        }
        else {
            self.u.assign(&g.t());
        }

        self.v.assign_eye(1.);

        self.do_decomp();
    }
    //
    /// Runs SVD of a specified matrix with a warm-start from the last SVD result.
    pub fn decomp_warm(&mut self, g: &Mat)
    {
        if !self.transposed {
            self.u.assign(&(g * &self.v));
        }
        else {
            self.u.assign(&(g.t() * &self.v));
        }

        self.do_decomp();
    }
    //
    /// Solves linear equations using the last SVD result.
    pub fn solve(&self, h: &Mat) -> Mat
    {
        let mut sinv = self.s.clone_diag();
        let (nrows, _) = self.s.size();

        for r in 0 .. nrows {
            let s = sinv[(r, r)];

            sinv[(r, r)] = if s.abs() < TOL_SINV {
                0.
            }
            else {
                1. / s
            }
        }

        if !self.transposed {
            &self.v * (sinv * (self.u.t() * h))
        }
        else {
            &self.u * (sinv * (self.v.t() * h))
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
        &svd.u * svd.s.clone_diag() * svd.v.t()
    }
    else {
        &svd.v * svd.s.clone_diag() * svd.u.t()
    };
    println!("mat reconstructed = {}", g);

    let g_size = g.size();
    let g_err = (g - mat).norm_p2sq() / ((g_size.0 * g_size.1) as FP);
    println!("g_err = {:e}", g_err);
    assert!(g_err < TOL_RMSE);

    //

    let mut utu = svd.u.t() * &svd.u;
    println!("u' * u = {}", utu);

    let utu_size = utu.size();
    utu.assign_by(|r, c| if r == c {Some(0.)} else {None});
    let utu_err = utu.norm_p2sq() / ((utu_size.0 * utu_size.1) as FP);
    println!("utu_err = {:e}", utu_err);
    assert!(utu_err < TOL_RMSE);

    //

    let mut vvt = &svd.v * svd.v.t();
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
