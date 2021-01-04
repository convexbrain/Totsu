// TODO: no blas/lapack

use crate::solver::{Cone, SolverParam, SolverError, LinAlg};
use crate::linalg::F64BLAS;

pub struct ConePSD<'a>
{
    work: &'a mut[f64],
}

impl<'a> ConePSD<'a>
{
    fn nvars_to_nrows(nvars: usize) -> usize
    {
        let nrows = (((8 * nvars + 1) as f64).sqrt() as usize - 1) / 2;
        assert_eq!(nrows * (nrows + 1) / 2, nvars);
        nrows
    }

    pub fn query_worklen(nvars_dual: usize) -> usize
    {
        let nrows = Self::nvars_to_nrows(nvars_dual);
        let len_a = nrows * nrows;
        let len_w = nrows;
        let len_z = nrows * nrows;
        len_a + len_w + len_z
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
        let nvars = x.len();
        let nrows = Self::nvars_to_nrows(nvars);

        if self.work.len() < (nrows * nrows) * 2 + nrows {
            return Err(SolverError::ConeFailure);
        }
    
        let (a, spl_work) = self.work.split_at_mut(nrows * nrows);
        let (mut w, spl_work) = spl_work.split_at_mut(nrows);
        let (mut z, _) = spl_work.split_at_mut(nrows * nrows);
        let mut m = 0;
    
        vec_to_mat(x, a);
    
        let n = nrows as i32;
        unsafe {
            lapacke::dsyevr(
                lapacke::Layout::RowMajor, b'V', b'V',
                b'L', n, a, n,
                0., f64::INFINITY, 0, 0, par.eps_zero,
                &mut m, &mut w,
                &mut z, n, &mut []);
        }
    
        for e in a.iter_mut() {
            *e = 0.;
        }
        for i in 0.. m as usize {
            let e = w[i];
            let (_, ref_z) = z.split_at(i);
            unsafe {
                cblas::dsyr(
                    cblas::Layout::RowMajor, cblas::Part::Lower,
                    n, e,
                    ref_z, n,
                    a, n);
            }
    
        }
    
        mat_to_vec(a, x);

        Ok(())
    }
}

// TODO: other cones
fn _proj_pos(x: &mut[f64])
{
    for e in x {
        *e = e.max(0.);
    }
}

fn _proj_o(x: &mut[f64])
{
    for e in x {
        *e = 0.;
    }
}

fn _proj_r(_x: &mut[f64])
{
    //
}

fn mat_to_vec(m: &mut[f64], v: &mut[f64])
{
    let l = v.len();
    let n = (m.len() as f64).sqrt() as usize;

    assert_eq!(m.len(), n * n);
    assert_eq!(n * (n + 1) / 2, l);

    // scale diagonals to match the resulted vector norm with the matrix norm multiplied by 0.5
    unsafe { cblas::dscal(n as i32, 0.5_f64.sqrt(), m, (n + 1) as i32) }

    let (_, mut ref_m) = m.split_at(0);
    let mut ref_v = v;

    // The vector is a symmetric matrix, packing the lower triangle by rows.
    for c in 0.. n {
        let (r, spl_m) = ref_m.split_at(n);
        ref_m = spl_m;
        let (rc, _) = r.split_at(c + 1);

        let (vc, spl_v) = ref_v.split_at_mut(c + 1);
        ref_v = spl_v;
        F64BLAS::copy(rc, vc);
    }

    assert!(ref_m.is_empty());
    assert!(ref_v.is_empty());
}

fn vec_to_mat(v: &[f64], m: &mut[f64])
{
    let l = v.len();
    let n = (m.len() as f64).sqrt() as usize;

    assert_eq!(m.len(), n * n);
    assert_eq!(n * (n + 1) / 2, l);

    let (_, mut ref_m) = m.split_at_mut(0);
    let mut ref_v = v;

    // The vector is a symmetric matrix, packing the lower triangle by rows.
    for c in 0.. n {
        let (r, spl_m) = ref_m.split_at_mut(n);
        ref_m = spl_m;
        let (rc, _) = r.split_at_mut(c + 1);

        let (vc, spl_v) = ref_v.split_at(c + 1);
        ref_v = spl_v;
        F64BLAS::copy(vc, rc);
    }

    assert!(ref_m.is_empty());
    assert!(ref_v.is_empty());

    // scale diagonals to match the resulted matrix norm with the vector norm multiplied by 2
    unsafe { cblas::dscal(n as i32, 2_f64.sqrt(), m, (n + 1) as i32) }
}


#[test]
fn test_cone1() {
    use float_eq::assert_float_eq;

    let ref_v = &[
         1.*0.7,
         2.,  3.*0.7,
         4.,  5.,  6.*0.7,
         7.,  8.,  9., 10.*0.7,
        11., 12., 13., 14., 15.*0.7,
    ];
    let ref_m = &[
         1.,  0.,  0.,  0.,  0.,
         2.,  3.,  0.,  0.,  0.,
         4.,  5.,  6.,  0.,  0.,
         7.,  8.,  9., 10.,  0.,
        11., 12., 13., 14., 15.,
    ];
    let mut v = ref_v.clone();
    let m = &mut[0.; 25];
    vec_to_mat(&mut v, m);
    assert_float_eq!(ref_m, m, abs_all <= 0.5);
    mat_to_vec(m, &mut v);
    assert_float_eq!(ref_v, &v, abs_all <= 1e-6);
}

#[test]
fn test_cone2() {
    use float_eq::assert_float_eq;

    let ref_x = &[
        5.,
        0., 0.,
    ];
    let x = &mut[
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
