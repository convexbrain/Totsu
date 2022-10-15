use num_traits::{Float, Zero, One};
use crate::solver::{Cone, SliceLike};
use crate::{LinAlgEx, ConeSOC, splitm_mut};

//

/// Rotated second-order (or quadratic) cone
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
/// 
/// \\[
/// \mathcal{Q}\_r^n =
/// \left\lbrace x \in \mathbb{R}^n
/// \ \middle|\ x_3^2+\cdots+x_n^2 \le 2x_1x_2, x_1\ge0, x_2\ge0
/// \right\rbrace
/// \\]
pub struct ConeRotSOC<L: LinAlgEx>
{
    soc: ConeSOC<L>,
}

impl<L: LinAlgEx> ConeRotSOC<L>
{
    /// Creates an instance.
    /// 
    /// Returns the [`ConeRotSOC`] instance.
    pub fn new() -> Self
    {
        ConeRotSOC {
            soc: ConeSOC::new(),
        }
    }
}

impl<L: LinAlgEx> Cone<L> for ConeRotSOC<L>
{
    fn proj(&mut self, dual_cone: bool, x: &mut L::Sl) -> Result<(), ()>
    {
        let f0 = L::F::zero();
        let f1 = L::F::one();
        let f2 = f1 + f1;
        let fsqrt2r = f2.sqrt() / f2;

        if x.len() > 0 {
            if x.len() == 1 {
                let r = x.get(0);
                x.set(0, r.max(f0));
            }
            else {
                {
                    splitm_mut!(x, (rs; 2));
                    L::add_sub(fsqrt2r, &mut rs);
                }

                self.soc.proj(dual_cone, x)?;

                {
                    splitm_mut!(x, (rs; 2));
                    L::add_sub(fsqrt2r, &mut rs);
                }
            }
        }
        Ok(())
    }

    fn product_group<G: Fn(&mut L::Sl) + Copy>(&self, dp_tau: &mut L::Sl, group: G)
    {
        group(dp_tau);
    }
}
