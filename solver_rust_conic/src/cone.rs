use crate::solver::{Cone, SolverParam, SolverError};
use crate::linalg::{LinAlgAux, F64BLAS};

pub struct ConePSD<'a>
{
    work: &'a mut[f64],
}

impl<'a> ConePSD<'a>
{
    pub fn query_worklen(nvars_dual: usize) -> usize
    {
        F64BLAS::proj_psd_worklen(nvars_dual)
    }

    pub fn new(work: &'a mut[f64]) -> Self
    {
        ConePSD {
            work
        }
    }
}

impl<'a> Cone<f64> for ConePSD<'a>
{
    fn proj(&mut self, par: &SolverParam<f64>, x: &mut[f64]) -> Result<(), SolverError>
    {
        if self.work.len() < F64BLAS::proj_psd_worklen(x.len()) {
            return Err(SolverError::ConeFailure);
        }

        F64BLAS::proj_psd(x, par.eps_zero, self.work);

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
    use float_eq::assert_float_eq;

    let ref_x = &[ // column-major, upper-triangle (seen as if transposed)
        5.,
        0., 0.,
    ];
    let x = &mut[ // column-major, upper-triangle (seen as if transposed)
        5.,
        0., -5.,
    ];
    assert_eq!(ConePSD::query_worklen(x.len()), 10);
    let w = &mut[0.; 10];
    let mut c = ConePSD::new(w);
    let p = SolverParam::default();
    c.proj(&p, x).unwrap();
    assert_float_eq!(ref_x, x, abs_all <= 1e-6);
}
