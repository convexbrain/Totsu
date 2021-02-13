/*!
Totsu ([凸](http://www.decodeunicode.org/en/u+51F8) in Japanese) means convex.

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

This crate for Rust provides **a first-order conic linear program solver**.

# Target problem

A common target problem is continuous scalar **convex optimization** such as LP, QP, QCQP, SOCP and SDP.
Each of those problems can be represented as a conic linear program:
\\[
\begin{array}{ll}
{\rm minimize} & c^T x \\\\
{\rm subject \ to} & A x + s = b \\\\
& s \in \mathcal{K},
\end{array}
\\]
where
* variables \\( x \in {\bf R}^n,\ s \in {\bf R}^m \\)
* \\( c \in {\bf R}^n \\) as an objective linear operator 
* \\( A \in {\bf R}^{m \times n} \\) and \\( b \in {\bf R}^m \\) as constraint linear operators 
* a nonempty, closed, convex cone \\( \mathcal{K} \\).

# Algorithm and design concepts

The author combines the two papers
[\[1\]](https://ieeexplore.ieee.org/abstract/document/6126441)
[\[2\]](https://arxiv.org/abs/1312.3039)
so that the homogeneous self-dual embedding matrix in [\[2\]](https://arxiv.org/abs/1312.3039)
is formed as a linear operator in [\[1\]](https://ieeexplore.ieee.org/abstract/document/6126441).

A core method [`solver::Solver::solve`] takes the following arguments:
* objective and constraint linear operators that implement [`operator::Operator`] trait and
* a projection onto a cone that implements [`cone::Cone`] trait.

Therefore solving a specific problem requires an implementation of those traits.
You can use pre-defined implementations (see [`problem`]),
as well as construct a user-defined tailored version for the reason of functionality and efficiency.
Modules [`operator`] and [`cone`] include several basic structs
that implement [`operator::Operator`] and [`cone::Cone`] trait.

Core linear algebra operations that [`solver::Solver`] requires
are abstracted by [`linalg::LinAlg`] trait,
while subtrait [`linalg::LinAlgEx`] is used for [`operator`],
[`cone`] and [`problem`] modules.
This crate includes two [`linalg::LinAlgEx`] implementors:
* [`linalg::FloatGeneric`] -
  `num::Float`-generic implementation (pure Rust but slow)
* [`linalg::F64LAPACK`] -
  `f64`-specific implementation using `cblas-sys` and `lapacke-sys`
  (you need a [BLAS/LAPACK source](https://github.com/blas-lapack-rs/blas-lapack-rs.github.io/wiki#sources) to link).

## Features

This crate can be used without the standard library (`#![no_std]`).
Use this in `Cargo.toml`:

```toml
[dependencies.totsu]
version = "0.7.0"
default-features = false
features = ["nostd"]
```

Some module and structs are not availale in this case:
* [`problem`]
* [`linalg::F64LAPACK`]
* [`logger::PrintLogger`]
* [`logger::IoWriteLogger`]
* [`operator::MatBuild`]

## Changelog

Changelog is available in [CHANGELOG.md](https://github.com/convexbrain/Totsu/blob/master/solver_rust_conic/CHANGELOG.md).

# Examples
## QP

```
use float_eq::assert_float_eq;
use totsu::prelude::*;
use totsu::operator::MatBuild;
use totsu::problem::ProbQP;

type LA = FloatGeneric<f64>;
type AMatBuild = MatBuild<LA, f64>;
type AProbQP = ProbQP<LA, f64>;
type ASolver = Solver<LA, f64>;

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
let rslt = s.solve(qp.problem(), NullLogger).unwrap();

assert_float_eq!(rslt.0[0..2], [2., 0.].as_ref(), abs_all <= 1e-3);
```

## Other Examples

You can find other [tests](https://github.com/convexbrain/Totsu/tree/master/solver_rust_conic/tests) of pre-defined problems.
More practical [examples](https://github.com/convexbrain/Totsu/tree/master/examples) are also available.

## References

1. T. Pock and A. Chambolle.
   "Diagonal preconditioning for first order primal-dual algorithms in convex optimization."
   2011 International Conference on Computer Vision. IEEE, 2011.
1. B. O’donoghue, et al.
   "Conic optimization via operator splitting and homogeneous self-dual embedding."
   Journal of Optimization Theory and Applications 169.3 (2016): 1042-1068.
1. N. Parikh and S. Boyd.
   "Proximal algorithms."
   Foundations and Trends in optimization 1.3 (2014): 127-239.
1. Mosek ApS.
   "MOSEK modeling cookbook."
   (2020).
1. S. Boyd and L. Vandenberghe.
   "Convex Optimization."
   (2004).
*/

#![no_std]
#[cfg(not(feature = "nostd"))]
extern crate std;

pub mod solver; // core, Float
mod utils;

pub mod linalg;
pub mod operator;
pub mod cone;
pub mod logger;

#[cfg(not(feature = "nostd"))]
pub mod problem;

/// Prelude
pub mod prelude // core, Float
{
    pub use super::solver::{Solver, SolverError, SolverParam};
    pub use super::linalg::{LinAlg, LinAlgEx, FloatGeneric};
    pub use super::operator::{Operator, MatType, MatOp};
    pub use super::cone::{Cone, ConeZero, ConeRPos, ConeSOC, ConePSD};
    pub use super::logger::NullLogger;
}

// TODO: more tests
