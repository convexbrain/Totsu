use num_traits::Float;
use core::marker::PhantomData;
use super::Cone;

//

/// Zero cone
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
/// 
/// \\[
/// \lbrace 0 \rbrace^n = \lbrace 0 \rbrace \times \cdots \times \lbrace 0 \rbrace =
/// \left\lbrace x \in \mathbb{R}^n
/// \ \middle|\ x=0
/// \right\rbrace
/// \\]
pub struct ConeZero<F>
{
    ph_f: PhantomData<F>,
}

impl<F> ConeZero<F>
{
    /// Creates an instance.
    /// 
    /// Returns [`ConeZero`] instance.
    pub fn new() -> Self
    {
        ConeZero {
            ph_f: PhantomData,
        }
    }
}

impl<F> Cone<F> for ConeZero<F>
where F: Float
{
    fn proj(&mut self, dual_cone: bool, x: &mut[F]) -> Result<(), ()>
    {
        if !dual_cone {
            for e in x {
                *e = F::zero();
            }
        }
        Ok(())
    }

    fn product_group<G: Fn(&mut[F]) + Copy>(&self, _dp_tau: &mut[F], _group: G)
    {
        // do nothing
    }
}
