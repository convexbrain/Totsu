use num_traits::Float;
use core::marker::PhantomData;
use crate::linalg::LinAlgEx;
use super::Cone;

//

/// Positive semidefinite cone
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
/// 
/// \\[
/// {\rm vec}({\bf S}\_+^k) =
/// \left\lbrace x \in {\bf R}^n,\ n=\frac12 k(k+1)
/// \ \middle|\ {\rm vec}^{-1}(x) \in {\bf S}\_+^k
/// \right\rbrace
/// \\]
/// \\( {\rm vec}(X) = (X_{11}\ \sqrt2 X_{12}\ X_{22}\ \sqrt2 X_{13}\ \sqrt2 X_{23}\ X_{33}\ \cdots)^T \\)
/// which extracts and scales the upper-triangular part of a symmetric matrix X in column-wise.
pub struct ConePSD<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    ph_l: PhantomData<L>,
    work: &'a mut[F],
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
    pub fn new(work: &'a mut[F]) -> Self
    {
        ConePSD {
            ph_l: PhantomData::<L>,
            work,
        }
    }
}

impl<'a, L, F> Cone<F> for ConePSD<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn proj(&mut self, _dual_cone: bool, eps_zero: F, x: &mut[F]) -> Result<(), ()>
    {
        if self.work.len() < L::proj_psd_worklen(x.len()) {
            return Err(());
        }

        L::proj_psd(x, eps_zero, self.work);

        Ok(())
    }

    fn product_group<G: Fn(&mut[F]) + Copy>(&self, dp_tau: &mut[F], group: G)
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
    let mut c = ConePSD::<L, _>::new(w);
    c.proj(false, 1e-12, x).unwrap();
    assert_float_eq!(ref_x, x, abs_all <= 1e-6);
}
