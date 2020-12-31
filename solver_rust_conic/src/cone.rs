// TODO: no blas/lapack

use crate::solver::Cone;
use crate::linalg::{scale, copy};

pub struct ConePSD<'a>
{
    work: &'a mut[f64],
    eps_zero: f64,
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

    pub fn new(work: &'a mut[f64], eps_zero: f64) -> Self
    {
        ConePSD {
            work, eps_zero
        }
    }
}

impl<'a> Cone for ConePSD<'a>
{
    fn proj(&mut self, x: &mut[f64])
    {
        let nvars = x.len();
        let nrows = Self::nvars_to_nrows(nvars);
    
        // TODO: error
        let (a, work) = self.work.split_at_mut(nrows * nrows);
    
        vec_to_mat(x, a);
    
        let mut m = 0;
        // TODO: error
        let (mut w, work) = work.split_at_mut(nrows);
        // TODO: error
        let (mut z, _work) = work.split_at_mut(nrows * nrows);
    
        let n = nrows as i32;
        unsafe {
            lapacke::dsyevr(
                lapacke::Layout::RowMajor, b'V', b'V',
                b'U', n, a, n,
                0., f64::INFINITY, 0, 0, self.eps_zero,
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
                    cblas::Layout::RowMajor, cblas::Part::Upper,
                    n, e,
                    ref_z, n,
                    a, n);
            }
    
        }
    
        mat_to_vec(a, x);
    }
}

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

fn mat_to_vec(m: &[f64], v: &mut[f64])
{
    let l = v.len();
    let n = (m.len() as f64).sqrt() as usize;
    // TODO: error when pub-ed
    assert_eq!(m.len(), n * n);
    assert_eq!(n * (n + 1) / 2, l);

    let mut ref_m = m;
    let mut ref_v = v;

    for c in 0.. n {
        // upper triangular elements of symmetric matrix vectorized in row-wise
        let (r, spl_m) = ref_m.split_at(n);
        ref_m = spl_m;
        let (_, rc) = r.split_at(c);

        let (vc, spl_v) = ref_v.split_at_mut(n - c);
        ref_v = spl_v;
        copy(rc, vc);

        let (_, vct) = vc.split_at_mut(1);
        scale(2_f64.sqrt(), vct);
    }

    assert!(ref_m.is_empty());
    assert!(ref_v.is_empty());
}

fn vec_to_mat(v: &[f64], m: &mut[f64])
{
    let l = v.len();
    let n = (m.len() as f64).sqrt() as usize;
    // TODO: error when pub-ed
    assert_eq!(m.len(), n * n);
    assert_eq!(n * (n + 1) / 2, l);

    let mut ref_m = m;
    let mut ref_v = v;

    for c in 0.. n {
        // upper triangular elements of symmetric matrix vectorized in row-wise
        let (r, spl_m) = ref_m.split_at_mut(n);
        ref_m = spl_m;
        let (_, rc) = r.split_at_mut(c);

        let (vc, spl_v) = ref_v.split_at(n - c);
        ref_v = spl_v;
        copy(vc, rc);

        let (_, rct) = rc.split_at_mut(1);
        scale(0.5_f64.sqrt(), rct);
    }

    assert!(ref_m.is_empty());
    assert!(ref_v.is_empty());
}
