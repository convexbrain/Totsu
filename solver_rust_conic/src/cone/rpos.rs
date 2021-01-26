use num::Float;
use core::marker::PhantomData;
use crate::solver::SolverError;
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
    fn proj(&mut self, _dual_cone: bool, _eps_zero: F, x: &mut[F]) -> Result<(), SolverError>
    {
        for e in x {
            *e = e.max(F::zero());
        }
        Ok(())
    }

    fn product_group(&self, _dp_tau: &mut[F], _group: fn(&mut[F]))
    {
        // do nothing
    }
}
