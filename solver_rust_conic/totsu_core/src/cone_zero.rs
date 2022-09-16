use num_traits::Zero;
use core::marker::PhantomData;
use crate::solver::{Cone, LinAlg};

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
pub struct ConeZero<L: LinAlg>
{
    ph_l: PhantomData<L>,
}

impl<L: LinAlg> ConeZero<L>
{
    /// Creates an instance.
    /// 
    /// Returns [`ConeZero`] instance.
    pub fn new() -> Self
    {
        ConeZero {
            ph_l: PhantomData,
        }
    }
}

impl<L: LinAlg> Cone<L> for ConeZero<L>
{
    fn proj(&mut self, dual_cone: bool, x: &mut L::Sl) -> Result<(), ()>
    {
        if !dual_cone {
            L::scale(L::F::zero(), x);
        }
        Ok(())
    }

    fn product_group<G: Fn(&mut L::Sl) + Copy>(&self, _dp_tau: &mut L::Sl, _group: G)
    {
        // do nothing
    }
}
