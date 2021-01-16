use num::Float;
use core::marker::PhantomData;
use crate::solver::SolverError;
use crate::linalg::LinAlg;
use super::Cone;

//

pub struct ConeSOC<L, F>
where L: LinAlg<F>, F: Float
{
    ph_l: PhantomData<L>,
    ph_f: PhantomData<F>,
}

impl<L, F> ConeSOC<L, F>
where L: LinAlg<F>, F: Float
{
    pub fn new() -> Self
    {
        ConeSOC {
            ph_l: PhantomData,
            ph_f: PhantomData,
        }
    }
}

impl<L, F> Cone<F> for ConeSOC<L, F>
where L: LinAlg<F>, F: Float
{
    fn proj(&mut self, _eps_zero: F, x: &mut[F]) -> Result<(), SolverError>
    {
        let f0 = F::zero();
        let f1 = F::one();
        let f2 = f1 + f1;

        if let Some((s, v)) = x.split_last_mut() {
            let norm_v = L::norm(v);

            if norm_v <= -*s {
                for e in v {
                    *e = f0;
                }
                *s = f0;
            }
            else if norm_v <= *s {
                // as they are
            }
            else {
                let alpha = (f1 + *s / norm_v) / f2;
                L::scale(alpha, v);
                *s = (norm_v + *s) / f2;
            }

        }
        Ok(())
    }
}
