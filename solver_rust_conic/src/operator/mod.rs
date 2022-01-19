//! Linear operator

use num_traits::Float;

/// Linear operator trait
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
/// 
/// Expresses a linear operator \\(K: \mathbb{R}^n \to \mathbb{R}^m\\) (or a matrix \\(K \in \mathbb{R}^{m \times n}\\)).
pub trait Operator<F: Float>
{
    /// Size of \\(K\\).
    /// 
    /// Returns a tuple of \\(m\\) and \\(n\\).
    fn size(&self) -> (usize, usize);

    /// Calculate \\(\alpha K x + \beta y\\).
    /// 
    /// * `alpha` is a scalar \\(\alpha\\).
    /// * `x` is a vector \\(x\\).
    ///   The length of `x` shall be \\(n\\).
    /// * `beta` is a scalar \\(\beta\\).
    /// * `y` is a vector \\(y\\) before entry,
    ///   \\(\alpha K x + \beta y\\) on exit.
    ///   The length of `y` shall be \\(m\\).
    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F]);

    /// Calculate \\(\alpha K^T x + \beta y\\).
    /// 
    /// * `alpha` is a scalar \\(\alpha\\).
    /// * `x` is a vector \\(x\\).
    ///   The length of `x` shall be \\(m\\).
    /// * `beta` is a scalar \\(\beta\\).
    /// * `y` is a vector \\(y\\) before entry,
    ///   \\(\alpha K^T x + \beta y\\) on exit.
    ///   The length of `y` shall be \\(n\\).
    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F]);

    fn abssum_cols(&self, tau: &mut[F]);

    fn abssum_rows(&self, sigma: &mut[F]);
}

//

mod matop;    // core, Float

#[cfg(feature = "std")]
mod matbuild; // std,  Float

pub use matop::*;

#[cfg(feature = "std")]
pub use matbuild::*;

#[cfg(feature = "std")]
pub mod reffn; // std,  Float
