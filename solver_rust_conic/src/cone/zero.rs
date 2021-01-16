use num::Float;
use core::marker::PhantomData;
use crate::solver::SolverError;
use super::Cone;

//

pub struct ConeZero<F>
{
    ph_f: PhantomData<F>,
}

impl<F> ConeZero<F>
{
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
    fn proj(&mut self, _eps_zero: F, x: &mut[F]) -> Result<(), SolverError>
    {
        for e in x {
            *e = F::zero();
        }
        Ok(())
    }
    fn dual_proj(&mut self, _eps_zero: F, _x: &mut[F]) -> Result<(), SolverError>
    {
        Ok(())
    }
}
