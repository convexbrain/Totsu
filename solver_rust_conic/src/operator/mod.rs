//! Linear operator

use num_traits::Float;
use crate::linalg::LinAlg;

/// Linear operator trait
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
/// 
/// Expresses a linear operator \\(K: \mathbb{R}^n \to \mathbb{R}^m\\) (or a matrix \\(K \in \mathbb{R}^{m \times n}\\)).
pub trait Operator<L: LinAlg<F>, F: Float>
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
    fn op(&self, alpha: F, x: &L::Vector, beta: F, y: &mut L::Vector);

    /// Calculate \\(\alpha K^T x + \beta y\\).
    /// 
    /// * `alpha` is a scalar \\(\alpha\\).
    /// * `x` is a vector \\(x\\).
    ///   The length of `x` shall be \\(m\\).
    /// * `beta` is a scalar \\(\beta\\).
    /// * `y` is a vector \\(y\\) before entry,
    ///   \\(\alpha K^T x + \beta y\\) on exit.
    ///   The length of `y` shall be \\(n\\).
    /// 
    /// The calculation shall be equivalent to the general reference implementation shown below.
    /// ```
    /// # use num_traits::Float;
    /// # use totsu::linalg::LinAlgEx;
    /// # use totsu::operator::Operator;
    /// # struct OpRef<L, F>(std::marker::PhantomData<L>, std::marker::PhantomData<F>);
    /// impl<L: LinAlgEx<F>, F: Float> Operator<F> for OpRef<L, F>
    /// {
    /// #   fn size(&self) -> (usize, usize) {(0, 0)}
    /// #   fn op(&self, alpha: F, x: &L::Vector, beta: F, y: &mut L::Vector) {}
    /// #   fn absadd_cols(&self, tau: &mut L::Vector) {}
    /// #   fn absadd_rows(&self, sigma: &mut L::Vector) {}
    ///     fn trans_op(&self, alpha: F, x: &L::Vector, beta: F, y: &mut L::Vector)
    ///     {
    ///         let f0 = F::zero();
    ///         let f1 = F::one();
    ///         let (m, n) = self.size();
    /// 
    ///         let mut col = std::vec![f0; m];
    ///         let mut row = std::vec![f0; n];
    /// 
    ///         let mut y_rest = y;
    ///         for c in 0.. n {
    ///             row[c] = f1;
    ///             self.op(f1, &row, f0, &mut col);
    ///             row[c] = f0;
    /// 
    ///             let (yc, y_lh) = y_rest.split_at_mut(1);
    ///             y_rest = y_lh;
    ///             L::transform_ge(true, m, 1, alpha, &col, x, beta, yc);
    ///         }
    ///     }
    /// }
    /// ```
    fn trans_op(&self, alpha: F, x: &L::Vector, beta: F, y: &mut L::Vector);

    /// Calculate \\(\left[ \tau_j + \sum_{i=0}^{m-1}|K_{ij}| \right]_{j=0,...,n-1}\\).
    /// 
    /// * `tau` is a vector \\(\tau\\) before entry,
    ///   \\(\left[ \tau_j + \sum_{i=0}^{m-1}|K_{ij}| \right]_{j=0,...,n-1}\\) on exit.
    ///   The length of `tau` shall be \\(n\\).
    /// 
    /// The calculation shall be equivalent to the general reference implementation shown below.
    /// ```
    /// # use num_traits::Float;
    /// # use totsu::linalg::LinAlg;
    /// # use totsu::operator::Operator;
    /// # struct OpRef<L, F>(std::marker::PhantomData<L>, std::marker::PhantomData<F>);
    /// impl<L: LinAlg<F>, F: Float> Operator<F> for OpRef<L, F>
    /// {
    /// #   fn size(&self) -> (usize, usize) {(0, 0)}
    /// #   fn op(&self, alpha: F, x: &L::Vector, beta: F, y: &mut L::Vector) {}
    /// #   fn trans_op(&self, alpha: F, x: &L::Vector, beta: F, y: &mut L::Vector) {}
    /// #   fn absadd_rows(&self, sigma: &mut L::Vector) {}
    ///     fn absadd_cols(&self, tau: &mut L::Vector)
    ///     {
    ///         let f0 = F::zero();
    ///         let f1 = F::one();
    ///         let (m, n) = self.size();
    /// 
    ///         let mut col = std::vec![f0; m];
    ///         let mut row = std::vec![f0; n];
    /// 
    ///         for (c, t) in tau.iter_mut().enumerate() {
    ///             row[c] = f1;
    ///             self.op(f1, &row, f0, &mut col);
    ///             row[c] = f0;
    /// 
    ///             *t = L::abssum(&col, 1) + *t;
    ///         }
    ///     }
    /// }
    /// ```
    fn absadd_cols(&self, tau: &mut L::Vector);

    /// Calculate \\(\left[ \sigma_i + \sum_{j=0}^{n-1}|K_{ij}| \right]_{i=0,...,m-1}\\).
    /// 
    /// * `sigma` is a vector \\(\sigma\\) before entry,
    ///   \\(\left[ \sigma_i + \sum_{j=0}^{n-1}|K_{ij}| \right]_{i=0,...,m-1}\\) on exit.
    ///   The length of `sigma` shall be \\(m\\).
    /// 
    /// The calculation shall be equivalent to the general reference implementation shown below.
    /// ```
    /// # use num_traits::Float;
    /// # use totsu::linalg::LinAlg;
    /// # use totsu::operator::Operator;
    /// # struct OpRef<L, F>(std::marker::PhantomData<L>, std::marker::PhantomData<F>);
    /// impl<L: LinAlg<F>, F: Float> Operator<F> for OpRef<L, F>
    /// {
    /// #   fn size(&self) -> (usize, usize) {(0, 0)}
    /// #   fn op(&self, alpha: F, x: &L::Vector, beta: F, y: &mut L::Vector) {}
    /// #   fn trans_op(&self, alpha: F, x: &L::Vector, beta: F, y: &mut L::Vector) {}
    /// #   fn absadd_cols(&self, tau: &mut L::Vector) {}
    ///     fn absadd_rows(&self, sigma: &mut L::Vector)
    ///     {
    ///         let f0 = F::zero();
    ///         let f1 = F::one();
    ///         let (m, n) = self.size();
    /// 
    ///         let mut col = std::vec![f0; m];
    ///         let mut row = std::vec![f0; n];
    /// 
    ///         for (r, s) in sigma.iter_mut().enumerate() {
    ///             col[r] = f1;
    ///             self.trans_op(f1, &col, f0, &mut row);
    ///             col[r] = f0;
    /// 
    ///             *s = L::abssum(&row, 1) + *s;
    ///         }
    ///     }
    /// }
    /// ```
    fn absadd_rows(&self, sigma: &mut L::Vector);
}

//

mod matop;    // core, Float

/* TODO
#[cfg(feature = "std")]
mod matbuild; // std,  Float
 */

pub use matop::*;

/* TODO
#[cfg(feature = "std")]
pub use matbuild::*;
 */
