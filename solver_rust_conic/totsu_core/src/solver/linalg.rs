//! Linear algebra

use num_traits::Float;
use crate::solver::SliceLike;

/// Linear algebra trait.
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
pub trait LinAlg
{
    /// Floating point data type used as scalars.
    type F: Float;

    /// Data type of slice of `F` used as vectors.
    type Sl: SliceLike<F=Self::F> + ?Sized;

    /// Calculate 2-norm (or euclidean norm) \\(\\|x\\|_2=\sqrt{\sum_i x_i^2}\\).
    /// 
    /// Returns the calculated norm.
    /// * `x` is a vector \\(x\\).
    fn norm(x: &Self::Sl) -> Self::F;

    /// Copy from a vector to another vector.
    /// 
    /// * `x` is a slice to copy.
    /// * `y` is a slice being copied to.
    ///   `x` and `y` shall have the same length.
    fn copy(x: &Self::Sl, y: &mut Self::Sl);

    /// Calculate \\(\alpha x\\).
    /// 
    /// * `alpha` is a scalar \\(\alpha\\).
    /// * `x` is a vector \\(x\\) before entry, \\(\alpha x\\) on exit.
    fn scale(alpha: Self::F, x: &mut Self::Sl);

    /// Calculate \\(\alpha x + y\\).
    /// 
    /// * `alpha` is a scalar \\(\alpha\\).
    /// * `x` is a vector \\(x\\).
    /// * `y` is a vector \\(y\\) before entry, \\(\alpha x + y\\) on exit.
    ///   `x` and `y` shall have the same length.
    fn add(alpha: Self::F, x: &Self::Sl, y: &mut Self::Sl);

    /// Calculate \\(s\mathbb{1} + y\\).
    /// 
    /// * `s` is a scalar \\(s\\).
    /// * `y` is a vector \\(y\\) before entry, \\(s\mathbb{1} + y\\) on exit.
    fn adds(s: Self::F, y: &mut Self::Sl);

    /// Calculate 1-norm (or sum of absolute values) \\(\\|x\\|_1=\sum_i |x_i|\\).
    /// 
    /// Returns the calculated norm.
    /// * `x` is a vector \\(x\\).
    /// * `incx` is spacing between elements of `x`
    fn abssum(x: &Self::Sl, incx: usize) -> Self::F;

    /// Calculate \\(\alpha D x + \beta y\\),
    /// where \\(D={\bf diag}(d)\\) is a diagonal matrix.
    /// 
    /// * `alpha` is a scalar \\(\alpha\\).
    /// * `mat` is a diagonal vector \\(d\\) of \\(D\\).
    /// * `x` is a vector \\(x\\).
    /// * `beta` is a scalar \\(\beta\\).
    /// * `y` is a vector \\(y\\) before entry, \\(\alpha D x + \beta y\\) on exit.
    ///   `mat`, `x` and `y` shall have the same length.
    fn transform_di(alpha: Self::F, mat: &Self::Sl, x: &Self::Sl, beta: Self::F, y: &mut Self::Sl);
}
