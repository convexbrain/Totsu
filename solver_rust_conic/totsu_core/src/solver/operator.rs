//! Linear operator

use crate::solver::LinAlg;

/// Linear operator trait
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
/// 
/// Expresses a linear operator \\(K: \mathbb{R}^n \to \mathbb{R}^m\\) (or a matrix \\(K \in \mathbb{R}^{m \times n}\\)).
pub trait Operator<L: LinAlg>
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
    fn op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl);

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
    /// # use num_traits::{Zero, One};
    /// # use totsu_core::solver::{SliceLike, Operator};
    /// # use totsu_core::{LinAlgEx, splitm, splitm_mut};
    /// # struct OpRef<L>(std::marker::PhantomData<L>);
    /// impl<L: LinAlgEx> Operator<L> for OpRef<L>
    /// {
    /// #   fn size(&self) -> (usize, usize) {(0, 0)}
    /// #   fn op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl) {}
    /// #   fn absadd_cols(&self, tau: &mut L::Sl) {}
    /// #   fn absadd_rows(&self, sigma: &mut L::Sl) {}
    ///     fn trans_op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    ///     {
    ///         let f0 = L::F::zero();
    ///         let f1 = L::F::one();
    ///         let (m, n) = self.size();
    /// 
    ///         let mut col_v = std::vec![f0; m];
    ///         let mut row_v = std::vec![f0; n];
    ///         let mut col = L::Sl::new_mut(&mut col_v);
    ///         let mut row = L::Sl::new_mut(&mut row_v);
    /// 
    ///         for c in 0.. n {
    ///             row.set(c, f1);
    ///             self.op(f1, &row, f0, &mut col);
    ///             row.set(c, f0);
    /// 
    ///             splitm_mut!(y, (_y_done; c), (yc; 1));
    ///             L::transform_ge(true, m, 1, alpha, &col, x, beta, &mut yc);
    ///         }
    ///     }
    /// }
    /// ```
    fn trans_op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl);

    /// Calculate \\(\left[ \tau_j + \sum_{i=0}^{m-1}|K_{ij}| \right]_{j=0,...,n-1}\\).
    /// 
    /// * `tau` is a vector \\(\tau\\) before entry,
    ///   \\(\left[ \tau_j + \sum_{i=0}^{m-1}|K_{ij}| \right]_{j=0,...,n-1}\\) on exit.
    ///   The length of `tau` shall be \\(n\\).
    /// 
    /// The calculation shall be equivalent to the general reference implementation shown below.
    /// ```
    /// # use num_traits::{Zero, One};
    /// # use totsu_core::solver::{SliceLike, LinAlg, Operator};
    /// # struct OpRef<L>(std::marker::PhantomData<L>);
    /// impl<L: LinAlg> Operator<L> for OpRef<L>
    /// {
    /// #   fn size(&self) -> (usize, usize) {(0, 0)}
    /// #   fn op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl) {}
    /// #   fn trans_op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl) {}
    /// #   fn absadd_rows(&self, sigma: &mut L::Sl) {}
    ///     fn absadd_cols(&self, tau: &mut L::Sl)
    ///     {
    ///         let f0 = L::F::zero();
    ///         let f1 = L::F::one();
    ///         let (m, n) = self.size();
    /// 
    ///         let mut col_v = std::vec![f0; m];
    ///         let mut row_v = std::vec![f0; n];
    ///         let mut col = L::Sl::new_mut(&mut col_v);
    ///         let mut row = L::Sl::new_mut(&mut row_v);
    /// 
    ///         for c in 0.. tau.len() {
    ///             row.set(c, f1);
    ///             self.op(f1, &row, f0, &mut col);
    ///             row.set(c, f0);
    /// 
    ///             let val_tau = tau.get(c) + L::abssum(&col, 1);
    ///             tau.set(c, val_tau);
    ///         }
    ///     }
    /// }
    /// ```
    fn absadd_cols(&self, tau: &mut L::Sl);

    /// Calculate \\(\left[ \sigma_i + \sum_{j=0}^{n-1}|K_{ij}| \right]_{i=0,...,m-1}\\).
    /// 
    /// * `sigma` is a vector \\(\sigma\\) before entry,
    ///   \\(\left[ \sigma_i + \sum_{j=0}^{n-1}|K_{ij}| \right]_{i=0,...,m-1}\\) on exit.
    ///   The length of `sigma` shall be \\(m\\).
    /// 
    /// The calculation shall be equivalent to the general reference implementation shown below.
    /// ```
    /// # use num_traits::{Zero, One};
    /// # use totsu_core::solver::{SliceLike, LinAlg, Operator};
    /// # struct OpRef<L>(std::marker::PhantomData<L>);
    /// impl<L: LinAlg> Operator<L> for OpRef<L>
    /// {
    /// #   fn size(&self) -> (usize, usize) {(0, 0)}
    /// #   fn op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl) {}
    /// #   fn trans_op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl) {}
    /// #   fn absadd_cols(&self, tau: &mut L::Sl) {}
    ///     fn absadd_rows(&self, sigma: &mut L::Sl)
    ///     {
    ///         let f0 = L::F::zero();
    ///         let f1 = L::F::one();
    ///         let (m, n) = self.size();
    /// 
    ///         let mut col_v = std::vec![f0; m];
    ///         let mut row_v = std::vec![f0; n];
    ///         let mut col = L::Sl::new_mut(&mut col_v);
    ///         let mut row = L::Sl::new_mut(&mut row_v);
    /// 
    ///         for r in 0.. sigma.len() {
    ///             col.set(r, f1);
    ///             self.trans_op(f1, &col, f0, &mut row);
    ///             col.set(r, f0);
    /// 
    ///             let val_sigma = sigma.get(r) + L::abssum(&row, 1);
    ///             sigma.set(r, val_sigma);
    ///         }
    ///     }
    /// }
    /// ```
    fn absadd_rows(&self, sigma: &mut L::Sl);
}
