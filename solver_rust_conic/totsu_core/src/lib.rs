/*!
Totsu ([凸](http://www.decodeunicode.org/en/u+51F8) in Japanese) means convex.

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>

This crate for Rust provides **a first-order conic linear program solver** for convex optimization.

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
* variables \\( x \in \mathbb{R}^n,\ s \in \mathbb{R}^m \\)
* \\( c \in \mathbb{R}^n \\) as an objective linear operator 
* \\( A \in \mathbb{R}^{m \times n} \\) and \\( b \in \mathbb{R}^m \\) as constraint linear operators 
* a nonempty, closed, convex cone \\( \mathcal{K} \\).

# Algorithm and design concepts

The author combines the two papers
[\[1\]](https://ieeexplore.ieee.org/abstract/document/6126441)
[\[2\]](https://arxiv.org/abs/1312.3039)
so that the homogeneous self-dual embedding matrix in [\[2\]](https://arxiv.org/abs/1312.3039)
is formed as a linear operator in [\[1\]](https://ieeexplore.ieee.org/abstract/document/6126441).

A core method [`solver::Solver::solve`] takes the following arguments:
* objective and constraint linear operators that implement [`solver::Operator`] trait and
* a projection onto a cone that implements [`solver::Cone`] trait.

Therefore solving a specific problem requires an implementation of those traits.
You can use pre-defined implementations: [`totsu` crate](https://crates.io/crates/totsu),
as well as construct a user-defined tailored version for the reason of functionality and efficiency.
This crate also contains several basic structs
that implement [`solver::Operator`] and [`solver::Cone`] trait.

Core linear algebra operations that [`solver::Solver`] requires
are abstracted by [`solver::LinAlg`] trait,
while subtrait [`LinAlgEx`] is used for the basic structs.
This crate contains a [`LinAlgEx`] implementor:
* [`FloatGeneric`] -
  `num::Float`-generic implementation (pure Rust but slow).

Other crates are also available:
* [`totsu_f64lapack` crate](https://crates.io/crates/totsu_f64lapack) -
  `f64`-specific implementation using BLAS/LAPACK.
* [`totsu_f32cuda` crate](https://crates.io/crates/totsu_f32cuda) -
  `f32`-specific implementation using CUDA(cuBLAS/cuSOLVER).

# Features

This crate can be used without the standard library (`#![no_std]`).
Use this in `Cargo.toml`:

```toml
[dependencies.totsu_core]
version = "0.*"
default-features = false
features = ["libm"]
```

# Examples

See [examples in GitHub](https://github.com/convexbrain/Totsu/tree/master/examples).

# References

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

pub mod solver;

//

mod linalg_ex;

pub use linalg_ex::*;

//

mod floatgeneric;

pub use floatgeneric::*;

//

mod matop;

pub use matop::*;

//

mod cone_zero;
mod cone_rpos;
mod cone_soc;
mod cone_rotsoc;
mod cone_psd;

pub use cone_zero::*;
pub use cone_rpos::*;
pub use cone_soc::*;
pub use cone_rotsoc::*;
pub use cone_psd::*;
