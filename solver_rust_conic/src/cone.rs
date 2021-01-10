use crate::solver::{Cone, SolverParam, SolverError};
use crate::linalg::LinAlgEx;
use core::marker::PhantomData;

pub struct ConePSD<'a, L>
where L: LinAlgEx<f64>
{
    _ph_l: PhantomData::<L>,
    work: &'a mut[f64],
}

impl<'a, L> ConePSD<'a, L>
where L: LinAlgEx<f64>
{
    pub fn query_worklen(nvars_dual: usize) -> usize
    {
        L::proj_psd_worklen(nvars_dual)
    }

    pub fn new(work: &'a mut[f64]) -> Self
    {
        ConePSD {
            _ph_l: PhantomData::<L>,
            work,
        }
    }
}

impl<'a, L> Cone<f64> for ConePSD<'a, L>
where L: LinAlgEx<f64>
{
    fn proj(&mut self, par: &SolverParam<f64>, x: &mut[f64]) -> Result<(), SolverError>
    {
        if self.work.len() < L::proj_psd_worklen(x.len()) {
            return Err(SolverError::ConeFailure);
        }

        L::proj_psd(x, par.eps_zero, self.work);

        Ok(())
    }
}

pub struct ConeRPos;

impl Cone<f64> for ConeRPos
{
    fn proj(&mut self, _par: &SolverParam<f64>, x: &mut[f64]) -> Result<(), SolverError>
    {
        for e in x {
            *e = e.max(0.);
        }
        Ok(())
    }
}

pub struct ConeZero;

impl Cone<f64> for ConeZero
{
    fn proj(&mut self, _par: &SolverParam<f64>, x: &mut[f64]) -> Result<(), SolverError>
    {
        for e in x {
            *e = 0.;
        }
        Ok(())
    }
    fn dual_proj(&mut self, _par: &SolverParam<f64>, _x: &mut[f64]) -> Result<(), SolverError>
    {
        Ok(())
    }
}


#[test]
fn test_cone1() {
    use crate::linalg::F64BLAS;
    use float_eq::assert_float_eq;
    
    type AConePSD<'a> = ConePSD<'a, F64BLAS>;

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
    let p = SolverParam::default();
    c.proj(&p, x).unwrap();
    assert_float_eq!(ref_x, x, abs_all <= 1e-6);
}
