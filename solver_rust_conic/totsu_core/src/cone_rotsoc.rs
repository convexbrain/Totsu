use num_traits::{Float, Zero, One};
use crate::solver::{Cone, LinAlg, SliceLike};
use crate::ConeSOC;

//

/// Rotated second-order (or quadratic) cone
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
/// 
/// \\[
/// \mathcal{Q}\_r^n =
/// \left\lbrace x \in \mathbb{R}^n
/// \ \middle|\ x_3^2+\cdots+x_n^2 \le 2x_1x_2, x_1\ge0, x_2\ge0
/// \right\rbrace
/// \\]
pub struct ConeRotSOC<L: LinAlg>
{
    soc: ConeSOC<L>,
}

impl<L: LinAlg> ConeRotSOC<L>
{
    /// Creates an instance.
    /// 
    /// Returns [`ConeRotSOC`] instance.
    pub fn new() -> Self
    {
        ConeRotSOC {
            soc: ConeSOC::new(),
        }
    }
}

impl<L: LinAlg> Cone<L> for ConeRotSOC<L>
{
    fn proj(&mut self, dual_cone: bool, x: &mut L::Sl) -> Result<(), ()>
    {
        let f0 = L::F::zero();
        let f1 = L::F::one();
        let f2 = f1 + f1;
        let fsqrt2 = f2.sqrt();

        if x.len() > 0 {
            if x.len() == 1 {
                let r = x.get(0);
                x.set(0, r.max(f0));
            }
            else {
                let r = x.get(0);
                let s = x.get(1);
                x.set(0, (r + s) / fsqrt2);
                x.set(1, (r - s) / fsqrt2);

                self.soc.proj(dual_cone, x)?;

                let r = x.get(0);
                let s = x.get(1);
                x.set(0, (r + s) / fsqrt2);
                x.set(1, (r - s) / fsqrt2);
            }
        }
        Ok(())
    }

    fn product_group<G: Fn(&mut L::Sl) + Copy>(&self, dp_tau: &mut L::Sl, group: G)
    {
        group(dp_tau);
    }
}
