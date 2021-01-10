use crate::solver::{Cone, SolverError, LinAlg};
use crate::linalgex::LinAlgEx;
use core::marker::PhantomData;
use num::Float;

//

pub struct ConePSD<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    _ph_l: PhantomData<L>,
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
            _ph_l: PhantomData::<L>,
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

pub struct ConeSOC<L, F>
where L: LinAlg<F>, F: Float
{
    _ph_l: PhantomData<L>,
    _ph_f: PhantomData<F>,
}

impl<L, F> ConeSOC<L, F>
where L: LinAlg<F>, F: Float
{
    pub fn new() -> Self
    {
        ConeSOC {
            _ph_l: PhantomData,
            _ph_f: PhantomData,
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

//

pub struct ConeRPos<F>
{
    _ph_f: PhantomData<F>,
}

impl<F> ConeRPos<F>
{
    pub fn new() -> Self
    {
        ConeRPos {
            _ph_f: PhantomData,
        }
    }
}

impl<F> Cone<F> for ConeRPos<F>
where F: Float
{
    fn proj(&mut self, _eps_zero: F, x: &mut[F]) -> Result<(), SolverError>
    {
        for e in x {
            *e = e.max(F::zero());
        }
        Ok(())
    }
}

//

pub struct ConeZero<F>
{
    _ph_f: PhantomData<F>,
}

impl<F> ConeZero<F>
{
    pub fn new() -> Self
    {
        ConeZero {
            _ph_f: PhantomData,
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

//

#[test]
fn test_cone1() {
    use crate::f64_lapack::F64LAPACK;
    use float_eq::assert_float_eq;
    
    type AConePSD<'a> = ConePSD<'a, F64LAPACK, f64>;

    let ref_x = &[ // column-major, upper-triangle (seen as if transposed)
        5.,
        0., 0.,
    ];
    let x = &mut[ // column-major, upper-triangle (seen as if transposed)
        5.,
        0., -5.,
    ];
    assert_eq!(AConePSD::query_worklen(x.len()), 10);
    let w = &mut[0.; 10];
    let mut c = AConePSD::new(w);
    c.proj(1e-12, x).unwrap();
    assert_float_eq!(ref_x, x, abs_all <= 1e-6);
}
