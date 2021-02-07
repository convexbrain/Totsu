//! Linear operator

use num::Float;

/// Linear operator trait
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
/// 
/// Expresses a matrix \\(A \in {\bf R}^{m \times n}\\) as a linear operator.
pub trait Operator<F: Float>
{
    /// Size of \\(A\\).
    /// 
    /// Returns a tuple of \\(m\\) and \\(n\\).
    fn size(&self) -> (usize, usize);

    /// Calculate \\(\alpha A x + \beta y\\).
    /// 
    /// * `alpha` is a scalar \\(\alpha\\).
    /// * `x` is a vector \\(x\\).
    ///   The length of `x` shall be \\(n\\).
    /// * `beta` is a scalar \\(\beta\\).
    /// * `y` is a vector \\(y\\) before entry,
    ///   \\(\alpha A x + \beta y\\) on exit.
    ///   The length of `y` shall be \\(m\\).
    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F]);

    /// Calculate \\(\alpha A^T x + \beta y\\).
    /// 
    /// * `alpha` is a scalar \\(\alpha\\).
    /// * `x` is a vector \\(x\\).
    ///   The length of `x` shall be \\(m\\).
    /// * `beta` is a scalar \\(\beta\\).
    /// * `y` is a vector \\(y\\) before entry,
    ///   \\(\alpha A^T x + \beta y\\) on exit.
    ///   The length of `y` shall be \\(n\\).
    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F]);
}

//

mod matop;    // core, Float
mod matbuild; // std,  Float

pub use matop::*;
pub use matbuild::*;
