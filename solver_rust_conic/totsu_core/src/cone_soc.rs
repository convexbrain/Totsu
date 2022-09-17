use num_traits::{Zero, One};
use core::marker::PhantomData;
use crate::solver::{Cone, LinAlg, SliceLike};

//

/// Second-order (or quadratic) cone
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
/// 
/// \\[
/// \mathcal{Q}^n =
/// \left\lbrace x \in \mathbb{R}^n
/// \ \middle|\ \sqrt{x_2^2+\cdots+x_n^2} \le x_1
/// \right\rbrace
/// \\]
pub struct ConeSOC<L: LinAlg>
{
    ph_l: PhantomData<L>,
}

impl<L: LinAlg> ConeSOC<L>
{
    /// Creates an instance.
    /// 
    /// Returns [`ConeSOC`] instance.
    pub fn new() -> Self
    {
        ConeSOC {
            ph_l: PhantomData,
        }
    }
}

impl<L: LinAlg> Cone<L> for ConeSOC<L>
{
    fn proj(&mut self, _dual_cone: bool, x: &mut L::Sl) -> Result<(), ()>
    {
        let f0 = L::F::zero();
        let f1 = L::F::one();
        let f2 = f1 + f1;

        if x.len() > 0 {
            let (mut s, mut v) = x.split_mut(1);

            let val_s = s.get(0);
            let norm_v = L::norm(&v);

            if norm_v <= -val_s {
                L::scale(f0, &mut v);
                s.set(0, f0);
            }
            else if norm_v <= val_s {
                // as they are
            }
            else {
                let alpha = (f1 + val_s / norm_v) / f2;
                L::scale(alpha, &mut v);
                s.set(0, (norm_v + val_s) / f2);
            }
        }

        Ok(())
    }

    fn product_group<G: Fn(&mut L::Sl) + Copy>(&self, dp_tau: &mut L::Sl, group: G)
    {
        group(dp_tau);
    }
}
