/*!
Totsu ([凸](http://www.decodeunicode.org/en/u+51F8) in Japanese) means convex.

<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML' async></script>

This crate for Rust provides a basic **primal-dual interior-point method** solver: [`PDIPM`](pdipm/struct.PDIPM.html).

# Target problem

A common target problem is continuous scalar **convex optimization** such as
LS, LP, QP, GP, QCQP and (approximately equivalent) SOCP.
More specifically,
\\[
\\begin{array}{ll}
{\\rm minimize} & f_{\\rm obj}(x) \\\\
{\\rm subject \\ to} & f_i(x) \\le 0 \\quad (i = 0, \\ldots, m - 1) \\\\
& A x = b,
\\end{array}
\\]
where
* variables \\( x \\in {\\bf R}^n \\)
* \\( f_{\\rm obj}: {\\bf R}^n \\rightarrow {\\bf R} \\), convex and twice differentiable
* \\( f_i: {\\bf R}^n \\rightarrow {\\bf R} \\), convex and twice differentiable
* \\( A \\in {\\bf R}^{p \\times n} \\), \\( b \\in {\\bf R}^p \\).

# Algorithm and design concepts

The overall algorithm is based on the reference:
*S. Boyd and L. Vandenberghe, "Convex Optimization",*
[http://stanford.edu/~boyd/cvxbook/](http://stanford.edu/~boyd/cvxbook/).

[`PDIPM`](pdipm/struct.PDIPM.html) has a core method [`solve`](pdipm/struct.PDIPM.html#method.solve)
which takes objective and constraint (derivative) functions as closures.
Therefore solving a specific problem requires a implementation of those closures.
You can use a pre-defined implementations (see [`predef`](predef/index.html)),
as well as construct a user-defined tailored version for the reason of functionality and efficiency.

This crate has no dependencies on other crates at all.
Necessary matrix operations are implemented in [`mat`](mat/index.html) and [`matsvd`](matsvd/index.html).

# Example: QP

```
use totsu::prelude::*;
use totsu::predef::*;

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

let pdipm = PDIPM::new();
let rslt = pdipm.solve_qp(std::io::sink(),
                          &mat_p, &vec_q,
                          &mat_g, &vec_h,
                          &mat_a, &vec_b).unwrap();

let exp = Mat::new_vec(n).set_iter(&[
    2., 0.
]);
assert!((&rslt - exp).norm_p2() < pdipm.eps, "rslt = {}", rslt);
```
*/

pub mod mat;
pub mod matsvd;
pub mod pdipm;
pub mod lp;
pub mod qp;
pub mod qcqp;
pub mod socp;

/// Prelude
pub mod prelude {
    pub use crate::mat::{Mat, FP};
    pub use crate::pdipm::PDIPM;
}

/// Pre-defined solvers
pub mod predef {
    pub use crate::lp::LP;
    pub use crate::qp::QP;
    pub use crate::qcqp::QCQP;
    pub use crate::socp::SOCP;
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

        let pdipm = PDIPM::new();
        let rslt = pdipm.solve_qp(std::io::sink(),
                                  &mat_p, &vec_q,
                                  &mat_g, &vec_h,
                                  &mat_a, &vec_b).unwrap();

        let exp = Mat::new_vec(n).set_iter(&[
            2., 0.
        ]);
        assert!((&rslt - exp).norm_p2() < pdipm.eps, "rslt = {}", rslt);
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

        let pdipm = PDIPM::new();
        let rslt = pdipm.solve_qcqp(std::io::sink(),
                                    &mat_p, &vec_q, &scl_r,
                                    &mat_a, &vec_b).unwrap();

        let exp = Mat::new_vec(n).set_iter(&[
            5., 4.
        ]);
        assert!((&rslt - exp).norm_p2() < pdipm.eps, "rslt = {}", rslt);
    }

    #[test]
    fn test_socp()
    {
        let n: usize = 2; // x0, x1
        let m: usize = 1;
        let p: usize = 0;
        let ni: usize = 2;

        let vec_f = Mat::new_vec(n).set_all(1.);
        let mut mat_g = vec![Mat::new(ni, n); m];
        let vec_h = vec![Mat::new_vec(ni); m];
        let vec_c = vec![Mat::new_vec(n); m];
        let mut scl_d = vec![0. as FP; m];

        mat_g[0].assign_iter(&[
            1., 0.,
            0., 1.
        ]);
        scl_d[0] = 1.41421356;

        let mat_a = Mat::new(p, n);
        let vec_b = Mat::new_vec(p);

        let pdipm = PDIPM::new();
        let rslt = pdipm.solve_socp(std::io::sink(),
                                    &vec_f,
                                    &mat_g, &vec_h, &vec_c, &scl_d,
                                    &mat_a, &vec_b).unwrap();

        let exp = Mat::new_vec(n).set_iter(&[
            -1., -1.
        ]);
        assert!((&rslt - exp).norm_p2() < pdipm.eps, "rslt = {}", rslt);
    }

    #[test]
    fn test_lp_infeas()
    {
        let n: usize = 1;
        let m: usize = 2;
        let p: usize = 0;

        let vec_c = Mat::new_vec(n).set_iter(&[
            1.
        ]);

        // x <= b, x >= c
        let mat_g = Mat::new(m, n).set_iter(&[
            1., -1.
        ]);
        let vec_h = Mat::new_vec(m).set_iter(&[
            -5., // b
            -(10.)  // -c
        ]);

        let mat_a = Mat::new(p, n);
        let vec_b = Mat::new_vec(p);

        let pdipm = PDIPM::new();
        let _rslt = pdipm.solve_lp(std::io::sink(),
                                   &vec_c,
                                   &mat_g, &vec_h,
                                   &mat_a, &vec_b).unwrap_err();
    }
}
