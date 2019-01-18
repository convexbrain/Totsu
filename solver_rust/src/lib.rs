pub mod mat;
pub mod matsvd;
pub mod pdipm;
pub mod qp;
pub mod qcqp;

pub mod prelude {
    pub use crate::mat::{Mat, FP};
    pub use crate::pdipm::PDIPM;
}

pub mod predef {
    pub use crate::qp::QP;
    pub use crate::qcqp::QCQP;
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use crate::predef::*;

    #[test]
    fn test_qp()
    {
        let n: usize = 2; // x0, x1
        let m: usize = 1;
        let p: usize = 0;

        // (1/2)(x - a)^2 + const
        let mat_p = Mat::new(n, n).set_iter(&[
            1., 0.,
            0., 1.
        ]);
        let vec_q = Mat::new_vec(n).set_iter(&[
            -(-1.), // -a0
            -(-2.)  // -a1
        ]);

        // 1 - x0/b0 - x1/b1 <= 0
        let mat_g = Mat::new(m, n).set_iter(&[
            -1. / 2., // -1/b0
            -1. / 3.  // -1/b1
        ]);
        let vec_h = Mat::new_vec(m).set_iter(&[
            -1.
        ]);

        let mat_a = Mat::new(p, n);
        let vec_b = Mat::new_vec(p);

        let rslt = PDIPM::new().solve_qp(std::io::sink(),
                                        &mat_p, &vec_q,
                                        &mat_g, &vec_h,
                                        &mat_a, &vec_b);
        println!("{}", rslt.unwrap());
    }

    #[test]
    fn test_qcqp()
    {
        let n: usize = 2; // x0, x1
        let m: usize = 1;
        let p: usize = 0;

        let mut mat_p = vec![Mat::new(n, n); m + 1];
        let mut vec_q = vec![Mat::new_vec(n); m + 1];
        let mut scl_r = vec![0. as FP; m + 1];

        // (1/2)(x - a)^2 + const
        mat_p[0].assign_iter(&[
            1., 0.,
            0., 1.
        ]);
        vec_q[0].assign_iter(&[
            -(5.), // -a0
            -(4.)  // -a1
        ]);

        // 1 - x0/b0 - x1/b1 <= 0
        vec_q[1].assign_iter(&[
            -1. / 2., // -1/b0
            -1. / 3.  // -1/b1
        ]);
        scl_r[1] = 1.;

        let mat_a = Mat::new(p, n);
        let vec_b = Mat::new_vec(p);

        let rslt = PDIPM::new().solve_qcqp(std::io::sink(),
                                           &mat_p, &vec_q, &scl_r,
                                           &mat_a, &vec_b);
        println!("{}", rslt.unwrap());
    }}
