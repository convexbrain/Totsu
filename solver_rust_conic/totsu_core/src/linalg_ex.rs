use crate::solver::{LinAlg, SliceLike};

/// Linear algebra extended subtrait
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
pub trait LinAlgEx: LinAlg + Clone
{
    /// Calculates \\(\alpha G x + \beta y\\).
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

    /// Calculates \\(\alpha S x + \beta y\\),
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

    /// Query of a length of work slice that [`LinAlgEx::map_eig`] requires.
    /// 
    /// Returns a length of work slice.
    /// * `n` is a number of rows and columns of \\(S\\) (see [`LinAlgEx::map_eig`]).
    fn map_eig_worklen(n: usize) -> usize;

    /// Applies a map to eigenvalues of a symmetric matrix \\(S\\) supplied in packed form.
    /// 
    /// 1. (optional) Scales diagonals of a given matrix \\(S\\).
    /// 1. Eigenvalue decomposition: \\(S \rightarrow V \mathbf{diag}(\lambda) V^T\\)
    /// 1. Applies a map to the eigenvalues: \\(\lambda \rightarrow \lambda'\\)
    /// 1. Reconstruct the matrix: \\(V \mathbf{diag'}(\lambda) V^T \rightarrow S'\\)
    /// 1. (optional) Inverted scaling to diagonals of \\(S'\\).
    /// 
    /// This routine is used for euclidean projection onto semidifinite matrix
    /// and taking a square root of the matrix.
    /// 
    /// * `mat` is the matrix \\(S\\) before entry, \\(S'\\) on exit.
    ///   It shall be stored in packed form (the upper-triangular part in column-wise).
    /// * `scale_diag` is the optional scaling factor to the diagonals.
    ///   `None` has the same effect as `Some(1.0)` but reduces computation.
    /// * `eps_zero` should be the same value as [`crate::solver::SolverParam::eps_zero`].
    /// * `work` slice is used for temporal variables.
    /// * `map` takes an eigenvalue and returns a modified eigenvalue.
    ///   Returning `None` has the same effect as no-modification but reduces computation.
    fn map_eig<M>(mat: &mut Self::Sl, scale_diag: Option<Self::F>, eps_zero: Self::F, work: &mut Self::Sl, map: M)
    where M: Fn(Self::F)->Option<Self::F>;

    // TODO: doc
    fn add_sub(alpha: Self::F, x: &mut Self::Sl)
    {
        assert_eq!(x.len(), 2);

        let x_mut = x.get_mut();
        let a = x_mut[0];
        let b = x_mut[1];
        x_mut[0] = (a + b) * alpha;
        x_mut[1] = (a - b) * alpha;
    }
}
