use num_traits::Float;
use crate::linalg::LinAlg;
use super::{Cone, ConeSOC};

//

/// Rotated second-order (or quadratic) cone
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
/// 
/// \\[
/// \mathcal{Q}\_r^n =
/// \left\lbrace x \in {\bf R}^n
/// \ \middle|\ x_3^2+\cdots+x_n^2 \le 2x_1x_2, x_1\ge0, x_2\ge0
/// \right\rbrace
/// \\]
pub struct ConeRotSOC<L, F>
where L: LinAlg<F>, F: Float
{
    soc: ConeSOC<L, F>,
}

impl<L, F> ConeRotSOC<L, F>
where L: LinAlg<F>, F: Float
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

impl<L, F> Cone<F> for ConeRotSOC<L, F>
where L: LinAlg<F>, F: Float
{
    fn proj(&mut self, dual_cone: bool, x: &mut[F]) -> Result<(), ()>
    {
        let f0 = F::zero();
        let f1 = F::one();
        let f2 = f1 + f1;
        let fsqrt2 = f2.sqrt();

        if x.len() > 0 {
            if x.len() == 1 {
                x[0] = x[0].max(f0);
            }
            else {
                let r = x[0];
                let s = x[1];
                x[0] = (r + s) / fsqrt2;
                x[1] = (r - s) / fsqrt2;

                self.soc.proj(dual_cone, x)?;

                let r = x[0];
                let s = x[1];
                x[0] = (r + s) / fsqrt2;
                x[1] = (r - s) / fsqrt2;
            }
        }
        Ok(())
    }

    fn product_group<G: Fn(&mut[F]) + Copy>(&self, dp_tau: &mut[F], group: G)
    {
        group(dp_tau);
    }
}
