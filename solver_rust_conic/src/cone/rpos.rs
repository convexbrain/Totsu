use num::Float;
use core::marker::PhantomData;
use super::Cone;

//

pub struct ConeRPos<F>
{
    ph_f: PhantomData<F>,
}

impl<F> ConeRPos<F>
{
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
    fn proj(&mut self, _dual_cone: bool, _eps_zero: F, x: &mut[F]) -> Result<(), ()>
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
