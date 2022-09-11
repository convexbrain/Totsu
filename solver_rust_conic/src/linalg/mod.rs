//! Linear algebra

use num_traits::Float;
use core::ops::{Deref, DerefMut, Drop};

//

// TODO: doc
pub trait SliceLike
{
    type F: Float;

    fn new(s: &[Self::F]) -> SliceRef<'_, Self>;
    fn new_mut(s: &mut[Self::F]) -> SliceMut<'_, Self>;

    fn split_at(&self, mid: usize) -> (SliceRef<'_, Self>, SliceRef<'_, Self>);
    fn split_at_mut(&mut self, mid: usize) -> (SliceMut<'_, Self>, SliceMut<'_, Self>);

    fn drop(&self);

    fn len(&self) -> usize;
    fn get_ref(&self) -> &[Self::F];
    fn get_mut(&mut self) -> &mut[Self::F];

    fn get(&self, idx: usize) -> Self::F
    {
        let (_, spl) = self.split_at(idx);
        let (ind, _) = spl.split_at(1);
        ind.get_ref()[0]
    }
    fn set(&mut self, idx: usize, val: Self::F)
    {
        let (_, mut spl) = self.split_at_mut(idx);
        let (mut ind, _) = spl.split_at_mut(1);
        ind.get_mut()[0] = val;
    }
}

// TODO: doc
#[derive(Debug)] // NOTE: Do not derive clone, or the functionality of SliceLike::drop may break.
pub struct SliceRef<'a, S: SliceLike + ?Sized>
{
    s: &'a S,
}

impl<'a, S: SliceLike + ?Sized> Deref for SliceRef<'a, S>
{
    type Target = S;
    fn deref(&self) -> &Self::Target {self.s}
}

impl<'a, S: SliceLike + ?Sized> Drop for SliceRef<'a, S>
{
    fn drop(&mut self) {
        self.s.drop();
    }
}

// TODO: doc
#[derive(Debug)]
pub struct SliceMut<'a, S: SliceLike + ?Sized>
{
    s: &'a mut S,
}

impl<'a, S: SliceLike + ?Sized> Deref for SliceMut<'a, S>
{
    type Target = S;
    fn deref(&self) -> &Self::Target {self.s}
}

impl<'a, S: SliceLike + ?Sized> DerefMut for SliceMut<'a, S>
{
    fn deref_mut(&mut self) -> &mut Self::Target {self.s}
}

impl<'a, S: SliceLike + ?Sized> Drop for SliceMut<'a, S>
{
    fn drop(&mut self) {
        self.s.drop();
    }
}

/// Linear algebra trait
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
pub trait LinAlg
{
    type F: Float;
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

/// Linear algebra extended subtrait
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
pub trait LinAlgEx: LinAlg
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

//

mod floatgeneric; // core, Float

#[cfg(feature = "f64lapack")]
mod f64lapack;    // core, f64(cblas/lapacke)

#[cfg(feature = "f32cuda")]
mod f32cuda;      // std, f32(rustacuda/cublas/cusolver)


pub use floatgeneric::*;

#[cfg(feature = "f64lapack")]
pub use f64lapack::*;

#[cfg(feature = "f32cuda")]
pub use f32cuda::*;
