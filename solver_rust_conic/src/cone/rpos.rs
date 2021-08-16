use num_traits::Float;
use core::marker::PhantomData;
use super::Cone;

//

/// Nonnegative orthant cone
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
/// 
/// \\[
/// \mathbb{R}\_+^n = \mathbb{R}\_+ \times \cdots \times \mathbb{R}\_+ =
/// \left\lbrace x \in \mathbb{R}^n
/// \ \middle|\ x_i \ge 0,\ i=1,\ldots,n
/// \right\rbrace
/// \\]
pub struct ConeRPos<F>
{
    ph_f: PhantomData<F>,
}

impl<F> ConeRPos<F>
{
    /// Creates an instance.
    /// 
    /// Returns [`ConeRPos`] instance.
    pub fn new() -> Self
    {
        ConeRPos {
            ph_f: PhantomData,
        }
    }
}

impl<F> Cone<F> for ConeRPos<F>
where F: Float
{
    fn proj(&mut self, _dual_cone: bool, x: &mut[F]) -> Result<(), ()>
    {
        for e in x {
            *e = e.max(F::zero());
        }
        Ok(())
    }

    fn product_group<G: Fn(&mut[F]) + Copy>(&self, _dp_tau: &mut[F], _group: G)
    {
        // do nothing
    }
}
