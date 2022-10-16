use num_traits::Zero;
use core::marker::PhantomData;
use crate::solver::{Cone, LinAlg};

//

/// Nonnegative orthant cone
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
/// 
/// \\[
/// \mathbb{R}\_+^n = \mathbb{R}\_+ \times \cdots \times \mathbb{R}\_+ =
/// \left\lbrace x \in \mathbb{R}^n
/// \ \middle|\ x_i \ge 0,\ i=1,\ldots,n
/// \right\rbrace
/// \\]
pub struct ConeRPos<L: LinAlg>
{
    ph_l: PhantomData<L>,
}

impl<L: LinAlg> ConeRPos<L>
{
    /// Creates an instance.
    /// 
    /// Returns the [`ConeRPos`] instance.
    pub fn new() -> Self
    {
        ConeRPos {
            ph_l: PhantomData,
        }
    }
}

impl<L: LinAlg> Cone<L> for ConeRPos<L>
{
    fn proj(&mut self, _dual_cone: bool, x: &mut L::Sl) -> Result<(), ()>
    {
        L::max(x, L::F::zero());
        Ok(())
    }

    fn product_group<G: Fn(&mut L::Sl) + Copy>(&self, _dp_tau: &mut L::Sl, _group: G)
    {
        // do nothing
    }
}
