/*!
Totsu ([凸](http://www.decodeunicode.org/en/u+51F8) in Japanese) means convex.

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>

This crate for Rust provides **BLAS/LAPACK linear algebra** operations for [`totsu`](https://crates.io/crates/totsu)/[`totsu_core`].

# General usage

* Refer to [`totsu`](https://crates.io/crates/totsu)/[`totsu_core`] first.
  This crate is designed to be used with them as an [`totsu_core::solver::LinAlg`]/[`totsu_core::LinAlgEx`] implementation.
* [`totsu_core::FloatGeneric<f64>`] can be replaced with [`F64LAPACK`].
* You need a [BLAS/LAPACK source](https://github.com/blas-lapack-rs/blas-lapack-rs.github.io/wiki#sources) to link
  because this crate uses `cblas-sys` and `lapacke-sys`.

# Examples

A simple QP problem:
\\[
\begin{array}{ll}
{\rm minimize} & {(x_0 - (-1))^2 + (x_1 - (-2))^2 \over 2} \\\\
{\rm subject \ to} & 1 - {x_0 \over 2} - {x_1 \over 3} <= 0
\end{array}
\\]

You will notice that a perpendicular drawn from \\((-1, -2)\\)
to the line \\(1 - {x_0 \over 2} - {x_1 \over 3} = 0\\) intersects
at point \\((2, 0)\\) which is the optimal solution of the problem.

```
use float_eq::assert_float_eq;
use totsu::prelude::*;
use totsu::*;

use totsu_f64lapack::F64LAPACK;
use intel_mkl_src as _; // Use any BLAS/LAPACK source crate.

//env_logger::init(); // Use any logger crate as `totsu` uses `log` crate.

type La = F64LAPACK;
type AMatBuild = MatBuild<La>;
type AProbQP = ProbQP<La>;
type ASolver = Solver<La>;

let n = 2; // x0, x1
let m = 1;
let p = 0;

// (1/2)(x - a)^2 + const
let mut sym_p = AMatBuild::new(MatType::SymPack(n));
sym_p[(0, 0)] = 1.;
sym_p[(1, 1)] = 1.;

let mut vec_q = AMatBuild::new(MatType::General(n, 1));
vec_q[(0, 0)] = -(-1.); // -a0
vec_q[(1, 0)] = -(-2.); // -a1

// 1 - x0/b0 - x1/b1 <= 0
let mut mat_g = AMatBuild::new(MatType::General(m, n));
mat_g[(0, 0)] = -1. / 2.; // -1/b0
mat_g[(0, 1)] = -1. / 3.; // -1/b1

let mut vec_h = AMatBuild::new(MatType::General(m, 1));
vec_h[(0, 0)] = -1.;

let mat_a = AMatBuild::new(MatType::General(p, n));

let vec_b = AMatBuild::new(MatType::General(p, 1));

let s = ASolver::new().par(|p| {
   p.max_iter = Some(100_000);
});
let mut qp = AProbQP::new(sym_p, vec_q, mat_g, vec_h, mat_a, vec_b, s.par.eps_zero);
let rslt = s.solve(qp.problem()).unwrap();

assert_float_eq!(rslt.0[0..2], [2., 0.].as_ref(), abs_all <= 1e-3);
```

## Other examples

See [examples in GitHub](https://github.com/convexbrain/Totsu/tree/master/examples).
*/

mod f64lapack;

pub use f64lapack::*;
