use num::Float;
use crate::linalg::LinAlg;
use super::{Cone, ConeSOC};

//

pub struct ConeRotSOC<L, F>
where L: LinAlg<F>, F: Float
{
    soc: ConeSOC<L, F>,
}

impl<L, F> ConeRotSOC<L, F>
where L: LinAlg<F>, F: Float
{
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
    fn proj(&mut self, dual_cone: bool, eps_zero: F, x: &mut[F]) -> Result<(), ()>
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
                let t = x[0];
                let s = x[1];
                x[0] = (t + s) / fsqrt2;
                x[1] = (t - s) / fsqrt2;

                self.soc.proj(dual_cone, eps_zero, x)?;

                let t = x[0];
                let s = x[1];
                x[0] = (t + s) / fsqrt2;
                x[1] = (t - s) / fsqrt2;
            }
        }
        Ok(())
    }

    fn product_group(&self, dp_tau: &mut[F], group: fn(&mut[F]))
    {
        group(dp_tau);
    }
}
