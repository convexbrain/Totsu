use num_traits::Float;
use core::marker::PhantomData;
use crate::linalg::{SliceMut, SliceLike, LinAlgEx};
use super::Cone;

//

/// Positive semidefinite cone
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
/// 
/// \\[
/// {\rm vec}(\mathcal{S}\_+^k) =
/// \left\lbrace x \in \mathbb{R}^n,\ n=\frac12 k(k+1)
/// \ \middle|\ {\rm vec}^{-1}(x) \in \mathcal{S}\_+^k
/// \right\rbrace
/// \\]
/// \\( {\rm vec}(X) = (X_{11}\ \sqrt2 X_{12}\ X_{22}\ \sqrt2 X_{13}\ \sqrt2 X_{23}\ X_{33}\ \cdots)^T \\)
/// which extracts and scales the upper-triangular part of a symmetric matrix X in column-wise.
pub struct ConePSD<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    ph_l: PhantomData<L>,
    work: SliceMut<'a, L::Slice>,
    eps_zero: F,
}

impl<'a, L, F> ConePSD<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    /// Query of a length of work slice.
    /// 
    /// Returns a length of work slice that [`ConePSD::new`] requires.
    /// * `nvars` is a number of variables, that is a length of `x` of [`ConePSD::proj`].
    pub fn query_worklen(nvars: usize) -> usize
    {
        L::proj_psd_worklen(nvars)
    }

    /// Creates an instance.
    /// 
    /// Returns [`ConePSD`] instance.
    /// * `work` slice is used for temporal variables in [`ConePSD::proj`].
    /// * `eps_zero` shall be the same value as [`crate::solver::SolverParam::eps_zero`].
    pub fn new(work: &'a mut[F], eps_zero: F) -> Self
    {
        ConePSD {
            ph_l: PhantomData::<L>,
            work: L::slice_mut(work),
            eps_zero,
        }
    }
}

impl<'a, L, F> Cone<L, F> for ConePSD<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn proj(&mut self, _dual_cone: bool, x: &mut L::Slice) -> Result<(), ()>
    {
        if self.work.len() < L::proj_psd_worklen(x.len()) {
            log::error!("work shortage: {} given < {} required", self.work.len(), L::proj_psd_worklen(x.len()));
            return Err(());
        }

        L::proj_psd(x, self.eps_zero, &mut self.work);

        Ok(())
    }

    fn product_group<G: Fn(&mut L::Slice) + Copy>(&self, dp_tau: &mut L::Slice, group: G)
    {
        group(dp_tau);
    }
}

//

#[test]
fn test_cone_psd1()
{
    use float_eq::assert_float_eq;
    use crate::linalg::FloatGeneric;

    type L = FloatGeneric<f64>;
    
    let ref_x = &[ // column-major, upper-triangle (seen as if transposed)
        5.,
        0., 0.,
    ];
    let x = &mut[ // column-major, upper-triangle (seen as if transposed)
        5.,
        0., -5.,
    ];
    assert!(ConePSD::<L, _>::query_worklen(x.len()) <= 10);
    let w = &mut[0.; 10];
    let mut c = ConePSD::<L, _>::new(w, 1e-12);
    c.proj(false, x).unwrap();
    assert_float_eq!(ref_x, x, abs_all <= 1e-6);
}
