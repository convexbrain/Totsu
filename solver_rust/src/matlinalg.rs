//! Matrix linear algebra

use super::mat::{Mat, MatSlice, FP, FP_EPSILON};
use super::operator::LinOp;

fn normalize(vec: MatSlice) -> (FP, Mat)
{
    assert_eq!(vec.size().1, 1);

    let n = vec.norm_p2();

    if n > 0. {
        (n, vec / n)
}
    else {
        (0., vec.clone_sz())
    }
}

fn solve_lsmr<L: LinOp>(lop_a: &L, vec_b: MatSlice) -> Mat
{
    const ATOL: FP = FP_EPSILON;
    const BTOL: FP = FP_EPSILON;
    const CONLIM: FP = 1. / FP_EPSILON;
    
    let (m, n) = lop_a.size();
    const ZERO: FP = 0.;
    const ONE: FP = 1.;

    assert_eq!(vec_b.size(), (m, 1));
    
    // 1. Initialize
    let (norm_b, mut u_1) = normalize(vec_b);
    //let beta_1 = norm_b;
    let (mut alpha_1, mut v_1) = normalize(lop_a.t_apply(&u_1).as_slice());
    let mut alpha_b1 = alpha_1;
    let mut zeta_b1 = alpha_1 * norm_b;
    let mut rho_0 = ONE;
    let mut rho_b0 = ONE;
    let mut c_b0 = ONE;
    let mut s_b0 = ZERO;
    let mut h_1 = v_1.clone_sz();
    let mut h_b0 = Mat::new_vec(n);
    let mut x_0 = Mat::new_vec(n);
    //
    let mut beta_u1 = norm_b;
    let mut beta_d0 = ZERO;
    let mut rho_d0 = ONE;
    let mut tau_tm1 = ZERO;
    let mut theta_t0 = ZERO;
    let mut zeta_0 = ZERO;
    //
    let mut sqnorm_b_1 = ZERO;
    let mut min_rho_b0 = rho_b0;
    let mut max_rho_b0 = rho_b0;

    if !(norm_b > 0.) || !(alpha_1 > 0.) {
        return x_0;
    }

    // 2. For k = 1, 2, 3 . . . , repeat steps 3–6.
    loop {
        // (indexing is shown as if k = 1)

        // 3. Continue the bidiagonalization
        let (beta_2, u_2) = normalize((lop_a.apply(&v_1) - alpha_1 * u_1).as_slice());
        let (alpha_2, v_2) = normalize((lop_a.t_apply(&u_2) - beta_2 * v_1).as_slice());

        // 4. Construct and apply rotation P_1
        let rho_1 = alpha_b1.hypot(beta_2);
        let c_1 = alpha_b1 / rho_1;
        let s_1 = beta_2 / rho_1;
        let theta_2 = s_1 * alpha_2;
        let alpha_b2 = c_1 * alpha_2;

        // 5. Construct and apply rotation P_b1
        let theta_b1 = s_b0 * rho_1;
        let rho_b1 = theta_2.hypot(c_b0 * rho_1);
        let c_b1 = c_b0 * rho_1 / rho_b1;
        let s_b1 = theta_2 / rho_b1;
        let zeta_1 = c_b1 * zeta_b1;
        let zeta_b2 = -s_b1 * zeta_b1;

        // 6. Update h, h_b, x
        let h_b1 = &h_1 - (theta_b1 * rho_1 / (rho_0 * rho_b0)) * h_b0;
        let x_1 = x_0 + (zeta_1 / (rho_1 * rho_b1)) * &h_b1;
        let h_2 = &v_2 - (theta_2 / rho_1) * h_1;

        // ||r_1|| 3. Apply rotation P_1
        let beta_h1 = c_1 * beta_u1;
        let beta_u2 = -s_1 * beta_u1;

        // ||r_1|| 4. If k >= 2, construct and apply rotation P_t0
        let rho_t0 = rho_d0.hypot(theta_b1);
        let c_t0 = rho_d0 / rho_t0;
        let s_t0 = theta_b1 / rho_t0;
        let theta_t1 = s_t0 * rho_b1;
        let rho_d1 = c_t0 * rho_b1;
        //let beta_t1 = c_t0 * beta_d0 + s_t0 * beta_h1;
        let beta_d1 = -s_t0 * beta_d0 + c_t0 * beta_h1;

        // ||r_1|| 5. Update t_t1 by forward substitution
        let tau_t0 = (zeta_0 - theta_t0 * tau_tm1) / rho_t0;
        let tau_d1 = (zeta_1 - theta_t1 * tau_t0) / rho_d1;

        // ||r_1|| 6. Form ||r_1||
        let norm_r_1 = beta_u2.hypot(beta_d1 - tau_d1);

        // (computing ||x_1||)
        let norm_x_1 = x_1.norm_p2();

        // (computing ||A||)
        let sqnorm_b_2 = sqnorm_b_1 + alpha_1.powi(2) + beta_2.powi(2);
        let norm_a = sqnorm_b_1.sqrt();

        // (computing ||cond(A)||)
        min_rho_b0 = min_rho_b0.min(rho_b0);
        max_rho_b0 = max_rho_b0.max(rho_b0);
        let c_rho = c_b0 * rho_1;
        let min_rho = min_rho_b0.min(c_rho);
        let max_rho = max_rho_b0.max(c_rho);
        let cond_a = max_rho / min_rho;

        // (computing ||A'r_1||)
        let norm_ar_1 = zeta_b2.abs();

        // (stopping criteria)
        let s1 = norm_r_1 <= BTOL * norm_b + ATOL * norm_a * norm_x_1;
        let s2 = norm_ar_1 <= ATOL * norm_a * norm_r_1;
        let s3 = cond_a >= CONLIM;

        if s1 || s2 || s3 {
            return x_1;
        }

        // (update variables)
        //
        //beta_1 = beta_2;
        u_1 = u_2;
        alpha_1 = alpha_2;
        v_1 = v_2;
        //
        rho_0 = rho_1;
        alpha_b1 = alpha_b2;
        //
        rho_b0 = rho_b1;
        c_b0 = c_b1;
        s_b0 = s_b1;
        zeta_0 = zeta_1;
        zeta_b1 = zeta_b2;
        //
        h_b0 = h_b1;
        x_0 = x_1;
        h_1 = h_2;
        //
        beta_u1 = beta_u2;
        theta_t0 = theta_t1;
        rho_d0 = rho_d1;
        beta_d0 = beta_d1;
        //
        tau_tm1 = tau_t0;
        //
        sqnorm_b_1 = sqnorm_b_2;
    }
}

/// Linear equation solver by LSMR
/// 
/// References
/// * [http://web.stanford.edu/group/SOL/software/lsmr/](http://web.stanford.edu/group/SOL/software/lsmr/)
/// * D. C.-L. Fong and M. A. Saunders, "LSMR: An iterative algorithm for sparse least-squares problems,"
///   SIAM J. Sci. Comput. 33:5, 2950-2971, published electronically Oct 27, 2011.
pub fn lin_solve<L: LinOp>(lop_a: &L, mat_b: &Mat) -> Mat
{
    let (_, xr) = lop_a.size();
    let (_, xc) = mat_b.size();
    let mut mat_x = Mat::new(xr, xc);

    for c in 0 .. xc {
        let vec_b = mat_b.col(c);
        let mut vec_x = mat_x.col_mut(c);
        vec_x.assign(&solve_lsmr(lop_a, vec_b));
    }

    mat_x
}

/// Finds dominant eigenvalue by power iteration
pub fn dom_eig(mat_a: &Mat) -> FP
{
    let (m, n) = mat_a.size();
    assert_eq!(m, n);

    if n == 0 {
        return 0.;
    }

    let mut v = Mat::new_vec(n).set_all(1.);
    let mut lambda = 0.;

    loop {
        let w = mat_a * &v;
        let lambda_n = v.prod(&w);

        if (lambda_n - lambda).abs() <= FP_EPSILON {
            return lambda_n;
        }

        let (_, v_n) = normalize(w.as_slice());
        v = v_n;
        lambda = lambda_n;
    }
}


#[test]
fn test_lsmr1()
{
    const TOL_RMSE: FP = 1.0 / (1u64 << 32) as FP;

    struct LOP {
        mat: Mat
    }
    impl LinOp for LOP {
        fn size(&self) -> (usize, usize) {
            self.mat.size()
        }
        fn mat(&self) -> Mat {
            self.mat.clone_sz()
        }
    }

    let lop = LOP {
        mat: Mat::new(2, 2).set_iter(&[
            1., 2.,
            3., 4.
        ])
    };

    let vec = Mat::new_vec(2).set_iter(&[
        5., 6.
    ]);

    let x = lin_solve(&lop, &vec);
    println!("x = {}", x);

    let h = lop.apply(&x);
    println!("vec reconstructed = {}", h);

    let h_size = h.size();
    let h_err = (h - vec).norm_p2sq() / ((h_size.0 * h_size.1) as FP);
    assert!(h_err < TOL_RMSE);
}

#[test]
fn test_lsmr2()
{
    const TOL_RMSE: FP = 1.0 / (1u64 << 32) as FP;

    struct LOP {
        mat: Mat
    }
    impl LinOp for LOP {
        fn size(&self) -> (usize, usize) {
            self.mat.size()
        }
        fn mat(&self) -> Mat {
            self.mat.clone_sz()
        }
    }

    let lop = LOP {
        mat: Mat::new(4, 4).set_iter(&[
            4.296e5,  0.000e0,  4.296e5,  0.000e0,
            0.000e0,  4.296e5,  4.296e5,  0.000e0,
            4.296e5,  4.296e5,  8.591e5,  1.000e0,
            0.000e0,  0.000e0,  1.000e0,  0.000e0
        ])
    };

    let vec = Mat::new_vec(4).set_iter(&[
        -9.460e1,
        -9.460e1,
        5.831e2,
        3.859e-3
    ]);

    let x = lin_solve(&lop, &vec);
    println!("x = {}", x);

    let h = lop.apply(&x);
    println!("vec reconstructed = {}", h);

    let h_size = h.size();
    let h_err = (h - vec).norm_p2sq() / ((h_size.0 * h_size.1) as FP);
    assert!(h_err < TOL_RMSE);
}
