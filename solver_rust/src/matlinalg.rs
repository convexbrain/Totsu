//! Matrix linear algebra

use super::mat::{Mat, MatSlice, FP, FP_EPSILON, FP_MINPOS};
use super::spmat::SpMat;


const ATOL: FP = FP_EPSILON;
const BTOL: FP = FP_EPSILON;
const CONLIM: FP = 1. / FP_EPSILON;
const INVTOL: FP = FP_MINPOS;

/// Solves sparse matrix linear equations by LSQR
/// 
/// References
/// * [http://web.stanford.edu/group/SOL/software/lsqr/](http://web.stanford.edu/group/SOL/software/lsqr/)
/// * C. C. Paige and M. A. Saunders, "LSQR: An algorithm for sparse linear equations and sparse least squares,"
///   TOMS 8(1), 43-71 (1982).
pub fn spsolve_lsqr(mat_a: &SpMat, mat_b: &Mat) -> Mat
{
    let (ar, xr) = mat_a.size();
    let (br, xc) = mat_b.size();

    assert_eq!(ar, br);

    let mut mat_x = Mat::new(xr, xc);

    for c in 0 .. xc {
        let vec_b = mat_b.col(c);
        let mut vec_x = mat_x.col_mut(c);
        vec_x.assign(&lsqr_vec(mat_a, vec_b));
    }

    mat_x
}

fn lsqr_vec(mat_a: &SpMat, vec_b: MatSlice) -> Mat
{
    let (m, n) = mat_a.size();
    const ZERO: FP = 0.;

    assert_eq!(vec_b.size(), (m, 1));
    
    let ki = Mat::new_vec(n).set_by(|r, _| {
        let k = mat_a.col(r).norm_p2();
        if k > INVTOL {
            k.recip()
        }
        else {
            1.
        }
    });
    
    // 1. Initialize
    let (beta_1, mut u_1) = normalize(vec_b);
    let (mut alpha_1, mut v_1) = normalize(ki.diag_mul(&mat_a.t().transform(&u_1)).as_slice());
    let mut w_1 = v_1.clone_sz();
    let mut x_0 = Mat::new_vec(n);
    let mut phi_b1 = beta_1;
    let mut rho_b1 = alpha_1;
    //
    let mut norm_r_0 = beta_1;
    let mut sqnorm_b_0 = ZERO;
    let mut sqnorm_d_0 = ZERO;
    let norm_b = beta_1;

    if !(beta_1 > 0.) || !(alpha_1 > 0.) {
        return x_0;
    }

    // 2. For i = 1, 2, 3 . . . , repeat steps 3–6.
    loop {
        // (indexing is shown as if i = 1)

        // 3. Continue the bidiagonalization
        let (beta_2, u_2) = normalize((mat_a.transform(&ki.diag_mul(&v_1)) - alpha_1 * u_1).as_slice());
        let (alpha_2, v_2) = normalize((ki.diag_mul(&mat_a.t().transform(&u_2)) - beta_2 * v_1).as_slice());

        // 4. Construct and apply next orthogonal transformation
        let rho_1 = rho_b1.hypot(beta_2);
        let c_1 = rho_b1 / rho_1;
        let s_1 = beta_2 / rho_1;
        let theta_2 = s_1 * alpha_2;
        let rho_b2 = -c_1 * alpha_2;
        let phi_1 = c_1 * phi_b1;
        let phi_b2 = s_1 * phi_b1;

        // 5. Update x, w
        let x_1 = &x_0 + (phi_1 / rho_1) * &w_1;
        let w_2 = &v_2 - (theta_2 / rho_1) * &w_1;

        // (computing ||r_1||)
        let norm_r_1 = norm_r_0 * s_1;

        // (computing ||A'r_1||)
        let norm_ar_1 = phi_b2 * alpha_2 * c_1.abs();

        // (computing ||x_1||)
        let norm_x_1 = x_1.norm_p2();

        // (computing ||A||)
        let sqnorm_b_1 = sqnorm_b_0 + alpha_1.powi(2) + beta_2.powi(2);
        let norm_a = sqnorm_b_1.sqrt();

        // (computing ||cond(A)||)
        let sqnorm_delta_1 = w_1.norm_p2sq() / rho_1.powi(2);
        let sqnorm_d_1 = sqnorm_d_0 + sqnorm_delta_1;
        let cond_a = (sqnorm_b_1 * sqnorm_d_1).sqrt();

        // (stopping criteria)
        let s1 = norm_r_1 <= BTOL * norm_b + ATOL * norm_a * norm_x_1;
        let s2 = norm_ar_1 <= ATOL * norm_a * norm_r_1;
        let s3 = cond_a >= CONLIM;

        if s1 || s2 || s3 {
            return ki.diag_mul(&x_1);
        }

        // (update variables)
        //
        //beta_1 = beta_2;
        u_1 = u_2;
        alpha_1 = alpha_2;
        v_1 = v_2;
        //
        rho_b1 = rho_b2;
        phi_b1 = phi_b2;
        //
        x_0 = x_1;
        w_1 = w_2;
        //
        norm_r_0 = norm_r_1;
        //
        sqnorm_b_0 = sqnorm_b_1;
        sqnorm_d_0 = sqnorm_d_1;
    }
}

fn normalize(vec: MatSlice) -> (FP, Mat)
{
    assert_eq!(vec.size().1, 1);

    let n = vec.norm_p2();

    if n > INVTOL {
        (n, vec / n)
    }
    else {
        (0., vec.clone_sz())
    }
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
fn test_lsqr1()
{
    const TOL_RMSE: FP = 1.0 / (1u64 << 32) as FP;

    let mat = SpMat::new(2, 2).set_iter(&[
        1., 2.,
        3., 4.
    ]);

    let vec = Mat::new_vec(2).set_iter(&[
        5., 6.
    ]);

    let x = spsolve_lsqr(&mat, &vec);
    println!("x = {}", x);

    let h = mat.transform(&x);
    println!("vec reconstructed = {}", h);

    let h_size = h.size();
    let h_err = (h - vec).norm_p2sq() / ((h_size.0 * h_size.1) as FP);
    assert!(h_err < TOL_RMSE);
}

#[test]
fn test_lsqr2()
{
    const TOL_RMSE: FP = 1.0 / (1u64 << 32) as FP;

    let mat = SpMat::new(4, 4).set_iter(&[
        4.296e5,  0.000e0,  4.296e5,  0.000e0,
        0.000e0,  4.296e5,  4.296e5,  0.000e0,
        4.296e5,  4.296e5,  8.591e5,  1.000e0,
        0.000e0,  0.000e0,  1.000e0,  0.000e0
    ]);

    let vec = Mat::new_vec(4).set_iter(&[
        -9.460e1,
        -9.460e1,
        5.831e2,
        3.859e-3
    ]);

    let x = spsolve_lsqr(&mat, &vec);
    println!("x = {}", x);

    let h = mat.transform(&x);
    println!("vec reconstructed = {}", h);

    let h_size = h.size();
    let h_err = (h - vec).norm_p2sq() / ((h_size.0 * h_size.1) as FP);
    assert!(h_err < TOL_RMSE);
}
