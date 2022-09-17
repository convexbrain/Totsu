// TODO: rename linalg_plus
// TODO: rearrange proj_psd and sqrt_spmat into one like eig_func

use crate::solver::LinAlg;

/// Linear algebra extended subtrait
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
pub trait LinAlgEx: LinAlg + Clone
{
    /// Calculate \\(\alpha G x + \beta y\\).
    /// 
    /// * If `transpose` is `true`, Calculate \\(\alpha G^T x + \beta y\\) instead.
    /// * `alpha` is a scalar \\(\alpha\\).
    /// * `n_row` is a number of rows of \\(G\\).
    /// * `n_col` is a number of columns of \\(G\\).
    /// * `mat` is a matrix \\(G\\), stored in column-major.
    ///   The length of `mat` shall be `n_row * n_col`.
    /// * `x` is a vector \\(x\\).
    ///   The length of `x` shall be `n_col` (or `n_row` if `transpose` is `true`).
    /// * `beta` is a scalar \\(\beta\\).
    /// * `y` is a vector \\(y\\) before entry,
    ///   \\(\alpha G x + \beta y\\) (or \\(\alpha G^T x + \beta y\\) if `transpose` is `true`) on exit.
    ///   The length of `y` shall be `n_row` (or `n_col` if `transpose` is `true`).
    fn transform_ge(transpose: bool, n_row: usize, n_col: usize, alpha: Self::F, mat: &Self::Sl, x: &Self::Sl, beta: Self::F, y: &mut Self::Sl);

    /// Calculate \\(\alpha S x + \beta y\\),
    /// where \\(S\\) is a symmetric matrix, supplied in packed form.
    /// 
    /// * `n` is a number of rows and columns of \\(S\\).
    /// * `alpha` is a scalar \\(\alpha\\).
    /// * `mat` is a matrix \\(S\\), stored in packed form (the upper-triangular part in column-wise).
    ///   The length of `mat` shall be `n * (n + 1) / 2`.
    /// * `x` is a vector \\(x\\).
    ///   The length of `x` shall be `n`.
    /// * `beta` is a scalar \\(\beta\\).
    /// * `y` is a vector \\(y\\) before entry, \\(\alpha S x + \beta y\\) on exit.
    ///   The length of `y` shall be `n`.
    fn transform_sp(n: usize, alpha: Self::F, mat: &Self::Sl, x: &Self::Sl, beta: Self::F, y: &mut Self::Sl);

    /// Query of a length of work slice that [`LinAlgEx::proj_psd`] requires.
    /// 
    /// Returns a length of work slice.
    /// * `sn` is a number of variables, that is a length of `x` of [`LinAlgEx::proj_psd`].
    fn proj_psd_worklen(sn: usize) -> usize;

    /// Euclidean projection \\(x\\) onto \\({\rm vec}(\mathcal{S}\_+^k)\\).
    /// 
    /// * `x` is \\(x\\), a vector to be projected before entry, and shall be replaced with the projected vector on exit.
    ///   The length of `x` shall be \\(\frac12k(k+1)\\)
    /// * `eps_zero` shall be the same value as [`crate::solver::SolverParam::eps_zero`].
    /// * `work` slice is used for temporal variables.
    fn proj_psd(x: &mut Self::Sl, eps_zero: Self::F, work: &mut Self::Sl);

    /// Query of a length of work slice that [`LinAlgEx::sqrt_spmat`] requires.
    /// 
    /// Returns a length of work slice.
    /// * `n` is a number of rows and columns of \\(S\\) (see [`LinAlgEx::sqrt_spmat`]).
    fn sqrt_spmat_worklen(n: usize) -> usize;

    /// Calculate \\(S^{\frac12}\\),
    /// where \\(S \in \mathcal{S}\_+^n\\), supplied in packed form.
    /// 
    /// * `mat` is a matrix \\(S\\) before entry, \\(S^{\frac12}\\) on exit.
    ///   It shall be stored in packed form (the upper-triangular part in column-wise).
    ///   The length of `mat` shall be \\(\frac12n(n+1)\\).
    /// * `eps_zero` should be the same value as [`crate::solver::SolverParam::eps_zero`].
    /// * `work` slice is used for temporal variables.
    fn sqrt_spmat(mat: &mut Self::Sl, eps_zero: Self::F, work: &mut Self::Sl);
}
