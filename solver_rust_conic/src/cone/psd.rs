use num::Float;
use core::marker::PhantomData;
use crate::solver::SolverError;
use crate::linalg::LinAlgEx;
use super::Cone;

//

pub struct ConePSD<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    ph_l: PhantomData<L>,
    work: &'a mut[F],
}

impl<'a, L, F> ConePSD<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    pub fn query_worklen(nvars_dual: usize) -> usize
    {
        L::proj_psd_worklen(nvars_dual)
    }

    pub fn new(work: &'a mut[F]) -> Self
    {
        ConePSD {
            ph_l: PhantomData::<L>,
            work,
        }
    }
}

impl<'a, L, F> Cone<F> for ConePSD<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn proj(&mut self, eps_zero: F, x: &mut[F]) -> Result<(), SolverError>
    {
        if self.work.len() < L::proj_psd_worklen(x.len()) {
            return Err(SolverError::ConeFailure);
        }

        L::proj_psd(x, eps_zero, self.work);

        Ok(())
    }
}

//

#[cfg(test)]
fn subtest_cone_psd1<L: LinAlgEx<f64>>()
{
    use float_eq::assert_float_eq;
    
    let ref_x = &[ // column-major, upper-triangle (seen as if transposed)
        5.,
        0., 0.,
    ];
    let x = &mut[ // column-major, upper-triangle (seen as if transposed)
        5.,
        0., -5.,
    ];
    assert!(ConePSD::<L, _>::query_worklen(x.len()) <= 10);
    let w = &mut[0.; 10];
    let mut c = ConePSD::<L, _>::new(w);
    c.proj(1e-12, x).unwrap();
    assert_float_eq!(ref_x, x, abs_all <= 1e-6);
}

#[test]
fn test_cone_psd1()
{
    use crate::linalg::{FloatGeneric, F64LAPACK};
    subtest_cone_psd1::<FloatGeneric<f64>>();
    subtest_cone_psd1::<F64LAPACK>();
}
