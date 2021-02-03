/*!
Totsu ([凸](http://www.decodeunicode.org/en/u+51F8) in Japanese) means convex.

<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML' async></script>

This crate for Rust provides **a first-order conic linear program solver**.

# Target problem

A common target problem is continuous scalar **convex optimization** such as
[LP](problem/struct.ProbLP.html),
[QP](problem/struct.ProbQP.html),
[QCQP](problem/struct.ProbQCQP.html),
[SOCP](problem/struct.ProbSOCP.html) and
[SDP](problem/struct.ProbSDP.html).
Each of those problems can be represented as a conic linear program:
\\[
\begin{array}{ll}
{\rm minimize} & c^T x \\\\
{\rm subject \ to} & A x + s = b \\\\
& s \in \mathcal{K},
\end{array}
\\]
where
* variables \\( x \in {\bf R}^n,\ s \in {\bf R}^m \\),
* \\( c \in {\bf R}^n \\) as an objective linear operator 
* \\( A \in {\bf R}^{m \times n} \\) and \\( b \in {\bf R}^m \\) as constraint linear operators 
* a nonempty, closed, convex cone \\( \mathcal{K} \\).

# Algorithm and design concepts

The author combines the two papers \[1\]\[2\]
so that the homogeneous self-dual embedding matrix in \[2\] is formed as a linear operator in \[1\].

A core method [`Solver::solve`](solver/struct.Solver.html#method.solve) takes the following arguments:
* objective and constraint linear operators that implement [`Operator`](operator/trait.Operator.html) trait and
* a projection onto a cone that implements [`Cone`](cone/trait.Cone.html) trait.

Therefore solving a specific problem requires an implementation of those traits.
You can use pre-defined implementations (see [`problem`](problem/index.html)),
as well as construct a user-defined tailored version for the reason of functionality and efficiency.
Modules [`operator`](operator/index.html) and [`cone`](cone/index.html) include several basic structs
that implement `Operator` and `Cone` trait.

Core linear algebra operations that `Solver` requires
are abstracted by [`LinAlg`](linalg/trait.LinAlg.html) trait.
Subtrait [`LinAlgEx`](linalg/trait.LinAlgEx.html) is used for `operator`, `cone` and `problem` modules on the other hand.
This crate includes two those implementors:
* [`FloatGeneric`](linalg/struct.FloatGeneric.html),
  `num::Float`-generic implementation (pure Rust but slow)
* [`F64LAPACK`](linalg/struct.F64LAPACK.html),
  `f64`-specific implementation using `cblas-sys` and `lapacke-sys`
  (you need a [BLAS/LAPACK source](https://github.com/blas-lapack-rs/blas-lapack-rs.github.io/wiki#sources) to link).

# Examples
## QP

*(TODO)*

## Other Examples

*(TODO)*

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
1. M. Andersen, et al.
   "Interior-point methods for large-scale cone programming."
   Optimization for machine learning 5583 (2011).
1. S. Boyd and L. Vandenberghe.
   "Convex Optimization."
   (2004).
*/

pub mod solver; // core, Float
mod utils;

pub mod linalg;
pub mod operator;
pub mod cone;
pub mod logger;
pub mod problem;

pub mod prelude // core, Float
{
    pub use super::solver::{Solver, SolverError, SolverParam};
    pub use super::linalg::{LinAlg, LinAlgEx, FloatGeneric};
    pub use super::operator::{Operator, MatType, MatOp};
    pub use super::cone::{Cone, ConeZero, ConeRPos, ConeSOC, ConePSD};
    pub use super::logger::NullLogger;
}

// TODO: no-std
// MUST TODO: doc, readme
// TODO: more tests
