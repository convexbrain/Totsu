// TODO: no blas/lapack

use num::Float;
use crate::solver::LinAlg;

pub trait LinAlgEx<F: Float>: LinAlg<F> + Clone
{
    // y = a*mat*x + b*y
    fn transform_ge(transpose: bool, n_row: usize, n_col: usize, alpha: f64, mat: &[f64], x: &[f64], beta: f64, y: &mut[f64]);
    // y = a*mat*x + b*y
    fn transform_sp(n: usize, alpha: f64, mat: &[f64], x: &[f64], beta: f64, y: &mut[f64]);
    fn proj_psd_worklen(sn: usize) -> usize;
    fn proj_psd(x: &mut[f64], eps_zero: f64, work: &mut[f64]);
}

#[derive(Clone)]
pub struct F64BLAS;

impl LinAlg<f64> for F64BLAS
{
    fn norm(x: &[f64]) -> f64
    {
        unsafe { cblas::dnrm2(x.len() as i32, x, 1) }
    }
    
    fn inner_prod(x: &[f64], y: &[f64]) -> f64
    {
        assert_eq!(x.len(), y.len());
    
        unsafe { cblas::ddot(x.len() as i32, x, 1, y, 1) }
    }
    
    fn copy(x: &[f64], y: &mut[f64])
    {
        assert_eq!(x.len(), y.len());
    
        unsafe { cblas::dcopy(x.len() as i32, x, 1, y, 1) }
    }
    
    fn scale(alpha: f64, x: &mut[f64])
    {
        unsafe { cblas::dscal(x.len() as i32, alpha, x, 1) }
    }
    
    fn add(alpha: f64, x: &[f64], y: &mut[f64])
    {
        assert_eq!(x.len(), y.len());
    
        unsafe { cblas::daxpy(x.len() as i32, alpha, x, 1, y, 1) }
    }
}

impl LinAlgEx<f64> for F64BLAS
{
    // y = a*mat*x + b*y
    fn transform_ge(transpose: bool, n_row: usize, n_col: usize, alpha: f64, mat: &[f64], x: &[f64], beta: f64, y: &mut[f64])
    {
        assert_eq!(mat.len(), n_row * n_col);

        let trans = if transpose {
            assert_eq!(x.len(), n_row);
            assert_eq!(y.len(), n_col);
    
            cblas::Transpose::Ordinary
        } else {
            assert_eq!(x.len(), n_col);
            assert_eq!(y.len(), n_row);

            cblas::Transpose::None
        };

        unsafe { cblas::dgemv(
            cblas::Layout::ColumnMajor, trans,
            n_row as i32, n_col as i32,
            alpha, mat, n_row as i32,
            x, 1,
            beta, y, 1
        ) }
    }

    // y = a*mat*x + b*y
    fn transform_sp(n: usize, alpha: f64, mat: &[f64], x: &[f64], beta: f64, y: &mut[f64])
    {
        assert_eq!(mat.len(), n * (n + 1) / 2);

        assert_eq!(x.len(), n);
        assert_eq!(y.len(), n);

        unsafe { cblas::dspmv(
            cblas::Layout::ColumnMajor, cblas::Part::Upper,
            n as i32,
            alpha, mat,
            x, 1,
            beta, y, 1
        ) }
    }

    fn proj_psd_worklen(sn: usize) -> usize
    {
        let n = (((8 * sn + 1) as f64).sqrt() as usize - 1) / 2;
        assert_eq!(n * (n + 1) / 2, sn);

        let len_a = n * n;
        let len_w = n;
        let len_z = n * n;

        len_a + len_w + len_z
    }

    fn proj_psd(x: &mut[f64], eps_zero: f64, work: &mut[f64])
    {
        let sn = x.len();

        let n = (((8 * sn + 1) as f64).sqrt() as usize - 1) / 2;
        assert_eq!(n * (n + 1) / 2, sn);
        assert!(work.len() >= Self::proj_psd_worklen(sn));

        let (a, spl_work) = work.split_at_mut(n * n);
        let (mut w, spl_work) = spl_work.split_at_mut(n);
        let (mut z, _) = spl_work.split_at_mut(n * n);
        let mut m = 0;
    
        vec_to_mat(x, a);
    
        unsafe {
            lapacke::dsyevr(
                lapacke::Layout::ColumnMajor, b'V', b'V',
                b'U', n as i32, a, n as i32,
                0., f64::INFINITY, 0, 0, eps_zero,
                &mut m, &mut w,
                &mut z, n as i32, &mut []);
        }
    
        for e in a.iter_mut() {
            *e = 0.;
        }
        for i in 0.. m as usize {
            let e = w[i];
            let (_, ref_z) = z.split_at(i * n);
            unsafe {
                cblas::dsyr(
                    cblas::Layout::ColumnMajor, cblas::Part::Upper,
                    n as i32, e,
                    ref_z, 1,
                    a, n as i32);
            }
    
        }
    
        mat_to_vec(a, x);
    }
}

fn vec_to_mat(v: &[f64], m: &mut[f64])
{
    let l = v.len();
    let n = (m.len() as f64).sqrt() as usize;

    assert_eq!(m.len(), n * n);
    assert_eq!(n * (n + 1) / 2, l);

    let (_, mut ref_m) = m.split_at_mut(0);
    let mut ref_v = v;

    // The vector is a symmetric matrix, packing the upper-triangle by columns.
    for c in 0.. n {
        let (col, spl_m) = ref_m.split_at_mut(n);
        ref_m = spl_m;
        let (cut, _) = col.split_at_mut(c + 1);

        let (v_cut, spl_v) = ref_v.split_at(c + 1);
        ref_v = spl_v;
        F64BLAS::copy(v_cut, cut);
    }

    assert!(ref_m.is_empty());
    assert!(ref_v.is_empty());

    // scale diagonals to match the resulted matrix norm with the vector norm multiplied by 2
    unsafe { cblas::dscal(n as i32, 2_f64.sqrt(), m, (n + 1) as i32) }
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

    // The vector is a symmetric matrix, packing the upper-triangle by columns.
    for c in 0.. n {
        let (col, spl_m) = ref_m.split_at(n);
        ref_m = spl_m;
        let (cut, _) = col.split_at(c + 1);

        let (v_cut, spl_v) = ref_v.split_at_mut(c + 1);
        ref_v = spl_v;
        F64BLAS::copy(cut, v_cut);
    }

    assert!(ref_m.is_empty());
    assert!(ref_v.is_empty());
}


#[test]
fn test_linalg1() {
    use float_eq::assert_float_eq;

    let ref_v = &[ // column-major, upper-triangle (seen as if transposed)
         1.*0.7,
         2.,  3.*0.7,
         4.,  5.,  6.*0.7,
         7.,  8.,  9., 10.*0.7,
        11., 12., 13., 14., 15.*0.7,
    ];
    let ref_m = &[ // column-major, upper-triangle (seen as if transposed)
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
