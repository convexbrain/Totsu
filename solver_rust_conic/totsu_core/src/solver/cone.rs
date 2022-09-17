//! Convex cone

use crate::solver::LinAlg;

/// Convex cone trait
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
pub trait Cone<L: LinAlg>
{
    /// Calculates \\(\Pi_\mathcal{K}(x)\\), that is euclidean projection of \\(x\\) onto the cone \\(\mathcal{K}\\).
    /// This is called by [`crate::solver::Solver::solve`] with passing dual variables as `x`.
    /// 
    /// Returns `Ok`, or `Err` if something fails.
    /// * If `dual_cone` is `true`, project onto the dual cone \\(\mathcal{K}^*\\).
    /// * `x` is \\(x\\), a vector to be projected before entry, and shall be replaced with the projected vector on exit.
    fn proj(&mut self, dual_cone: bool, x: &mut L::Sl) -> Result<(), ()>;

    /// Performs grouping for a diagonal preconditioning vector according to the cone \\(\mathcal{K}\\).
    /// 
    /// When \\(\mathcal{K}=\mathcal{K}^{n_1}\times\cdots\times\mathcal{K}^{n_q}\\),
    /// split `dp_tau` into \\(n_1,\ldots,n_q\\) elements groups and call `group` for each.
    /// It is no need to call `group` for the group of \\(n_i=1\\).
    /// 
    /// * `dp_tau` is a diagonal preconditioning vector to be grouped before entry,
    ///   and shall be replaced with the grouped vector on exit.
    /// * `group` is a grouping function provided by [`crate::solver::Solver::solve`].
    fn product_group<G: Fn(&mut L::Sl) + Copy>(&self, dp_tau: &mut L::Sl, group: G);
}
