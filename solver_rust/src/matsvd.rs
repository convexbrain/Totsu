use super::mat::{Mat, FP, FP_EPSILON, FP_MIN}; // TODO: prelude

const TOL_CNV2: FP = FP_EPSILON * FP_EPSILON;
const TOL_DIV0: FP = FP_MIN;
const TOL_SINV: FP = FP_EPSILON;

#[derive(Debug)]
pub struct MatSVD<'a>
{
    transposed: bool,
    //
    u: Mat<'a>,
    s: Mat<'a>,
    v: Mat<'a>
}

impl<'a> MatSVD<'a>
{
    pub fn new(g: Mat<'a>) -> MatSVD<'a>
    {
        let (nrows, ncols) = g.size();
        let transposed = nrows < ncols;

        let (u_nrows, u_ncols) = if !transposed {
            (nrows, ncols)
        }
        else {
            (ncols, nrows)
        };

        let mut svd = MatSVD {
            transposed,
            u: Mat::new(u_nrows, u_ncols),
            s: Mat::new1(u_ncols),
            v: Mat::new(u_ncols, u_ncols)
        };

        // TODO: re-initialize

        if !transposed {
            svd.u.assign(&g);
        }
        else {
            svd.u.assign(&g.t());
        }

        svd.v = svd.v.set_eye();

        svd
    }
    //
    fn apply_jacobi_rot(&mut self, c1: usize, c2: usize) -> bool
    {
        let a = (self.u.col(c1).t() * self.u.col(c1))[(0, 0)];
        let b = (self.u.col(c2).t() * self.u.col(c2))[(0, 0)];
        let d = (self.u.col(c1).t() * self.u.col(c2))[(0, 0)];

        let converged = d * d <= TOL_CNV2 * a * b;

        if !converged {
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
        }
        
        converged
    }
    //
    fn norm_singular(&mut self)
    {
        let (_, n) = self.u.size();

        for i in 0 .. n {
            let s = FP::sqrt((self.u.col(i).t() * self.u.col(i))[(0, 0)]);
            self.s[(i, 0)] = s;

            if (-TOL_DIV0 < s) && (s < TOL_DIV0) {
                continue;
            }

            let tmp = self.u.col(i) / s;

            self.u.col_mut(i).assign(&tmp);
        }
    }
    //
    pub fn decomp(&mut self)
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
    pub fn solve(&self, h: &Mat) -> Mat
    {
        let mut sinv = self.s.diag();
        let (nrows, _) = self.s.size();

        for r in 0 .. nrows {
            let s = sinv[(r, r)];
            sinv[(r, r)] = if (-TOL_SINV < s) && (s < TOL_SINV) {
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
}

#[test]
fn test_decomp()
{
    const TOL_RMSE: FP = 1.0 / (1u64 << 32) as FP;

    let mat = Mat::new(4, 4).set_by(|_| {rand::random()});
    println!("mat = {}", mat);

    let mut svd = MatSVD::new(mat.clone());

    svd.decomp();

    //

    let g = if !svd.transposed {
        &svd.u * svd.s.diag() * svd.v.t()
    }
    else {
        &svd.v * svd.s.diag() * svd.u.t()
    };
    println!("mat reconstructed = {}", g);

    let g_size = g.size();
    let g_err = (g - mat).sq_sum() / ((g_size.0 * g_size.1) as FP);
    assert!(g_err < TOL_RMSE);

    //

    let mut utu = svd.u.t() * &svd.u;
    println!("u' * u = {}", utu);

    let utu_size = utu.size();
    for k in 0 .. utu_size.0 {
        utu[(k, k)] = 0.;
    }
    let utu_err = utu.sq_sum() / ((utu_size.0 * utu_size.1) as FP);
    assert!(utu_err < TOL_RMSE);

    //

    let mut vvt = &svd.v * svd.v.t();
    println!("v * v' = {}", vvt);

    let vvt_size = vvt.size();
    for k in 0 .. vvt_size.0 {
        vvt[(k, k)] = 0.;
    }
    let vvt_err = vvt.sq_sum() / ((vvt_size.0 * vvt_size.1) as FP);
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

    let mut svd = MatSVD::new(mat.clone());

    svd.decomp();

    let vec = Mat::new1(2).set_iter(&[
        5., 6.
    ]);

    let x = svd.solve(&vec);
    println!("x = {}", x);

    let h = &mat * &x;
    println!("vec reconstructed = {}", h);

    let h_size = h.size();
    let h_err = (h - vec).sq_sum() / ((h_size.0 * h_size.1) as FP);
    assert!(h_err < TOL_RMSE);
}
