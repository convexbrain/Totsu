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
    fn proj(&mut self, dual_cone: bool, _eps_zero: F, x: &mut[F]) -> Result<(), SolverError>
    {
        if !dual_cone {
            for e in x {
                *e = F::zero();
            }
        }
        Ok(())
    }

    fn product_group(&self, _dp_tau: &mut[F], _group: fn(&mut[F]))
    {
        // do nothing
    }
}
