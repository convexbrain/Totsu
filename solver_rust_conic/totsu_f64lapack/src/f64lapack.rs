use totsu_core::solver::LinAlg;
use totsu_core::LinAlgEx;

//

/// `f64`-specific [`LinAlgEx`] implementation using `cblas-sys` and `lapacke-sys`
/// 
/// You need a [BLAS/LAPACK source](https://github.com/blas-lapack-rs/blas-lapack-rs.github.io/wiki#sources) to link.
#[derive(Clone)]
pub struct F64LAPACK;

impl LinAlg for F64LAPACK
{
    type F = f64;
    type Sl = [f64];
    
    fn norm(x: &[f64]) -> f64
    {
        unsafe { cblas::dnrm2(x.len() as i32, x, 1) }
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

    fn adds(s: f64, y: &mut[f64])
    {
        let one = &[1.];

        unsafe { cblas::daxpy(y.len() as i32, s, one, 0, y, 1) }
    }
    
    fn abssum(x: &[f64], incx: usize) -> f64
    {
        if incx == 0 {
            0.
        }
        else {
            unsafe { cblas::dasum(((x.len() + (incx - 1)) / incx) as i32, x, incx as i32) }
        }
    }

    fn transform_di(alpha: f64, mat: &[f64], x: &[f64], beta: f64, y: &mut[f64])
    {
        assert_eq!(mat.len(), x.len());
        assert_eq!(mat.len(), y.len());

        unsafe { cblas::dsbmv(
            cblas::Layout::ColumnMajor, cblas::Part::Upper,
            mat.len() as i32, 0,
            alpha, mat, 1,
            x, 1,
            beta, y, 1
        ) }
    }
}

//

fn eig_func<E>(a: &mut[f64], n: usize, eps_zero: f64, wz: &mut[f64], func: E)
where E: Fn(f64)->Option<f64>
{
    let (w, rest) = wz.split_at_mut(n);
    let (z, _) = rest.split_at_mut(n * n);
    let mut m = 0;

    unsafe {
        lapacke::dsyevr(
            lapacke::Layout::ColumnMajor, b'V', b'V',
            b'U', n as i32, a, n as i32,
            0., f64::INFINITY, 0, 0, eps_zero,
            &mut m, w,
            z, n as i32, &mut []);
    }

    F64LAPACK::scale(0., a);

    for i in 0.. m as usize {
        if let Some(e) = func(w[i]) {
            let (_, ref_z) = z.split_at(i * n);
            unsafe {
                cblas::dsyr(
                    cblas::Layout::ColumnMajor, cblas::Part::Upper,
                    n as i32, e,
                    ref_z, 1,
                    a, n as i32);
            }
        }
    }
}

fn eig_func_worklen(n: usize) -> usize
{
    let len_w = n;
    let len_z = n * n;

    len_w + len_z
}

//

impl LinAlgEx for F64LAPACK
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

    fn map_eig_worklen(n: usize) -> usize
    {
        let len_a = n * n;

        len_a + eig_func_worklen(n)
    }

    fn map_eig<M>(mat: &mut[f64], scale_diag: Option<f64>, eps_zero: f64, work: &mut[f64], map: M)
    where M: Fn(f64)->Option<f64>
    {
        let sn = mat.len();

        let n = (((8 * sn + 1) as f64).sqrt() as usize - 1) / 2;
        assert_eq!(n * (n + 1) / 2, sn);

        assert!(work.len() >= Self::map_eig_worklen(n));

        let (a, rest) = work.split_at_mut(n * n);
        let (wz, _) = rest.split_at_mut(n + n * n);

        vec_to_mat(mat, a, scale_diag);
    
        eig_func(a, n, eps_zero, wz, map);

        mat_to_vec(a, mat, scale_diag);
    }
}

//

fn vec_to_mat(v: &[f64], m: &mut[f64], scale: Option<f64>)
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
        F64LAPACK::copy(v_cut, cut);
    }

    assert!(ref_m.is_empty());
    assert!(ref_v.is_empty());

    if let Some(scl) = scale {
        // scale diagonals
        unsafe { cblas::dscal(n as i32, scl, m, (n + 1) as i32) }
    }
}

fn mat_to_vec(m: &mut[f64], v: &mut[f64], scale: Option<f64>)
{
    let l = v.len();
    let n = (m.len() as f64).sqrt() as usize;

    assert_eq!(m.len(), n * n);
    assert_eq!(n * (n + 1) / 2, l);

    if let Some(scl) = scale {
        // revert scaled diagonals
        unsafe { cblas::dscal(n as i32, scl.recip(), m, (n + 1) as i32) }
    }

    let (_, mut ref_m) = m.split_at(0);
    let mut ref_v = v;

    // The vector is a symmetric matrix, packing the upper-triangle by columns.
    for c in 0.. n {
        let (col, spl_m) = ref_m.split_at(n);
        ref_m = spl_m;
        let (cut, _) = col.split_at(c + 1);

        let (v_cut, spl_v) = ref_v.split_at_mut(c + 1);
        ref_v = spl_v;
        F64LAPACK::copy(cut, v_cut);
    }

    assert!(ref_m.is_empty());
    assert!(ref_v.is_empty());
}

//

#[cfg(test)]
use intel_mkl_src as _;

#[test]
fn test_f64lapack1()
{
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
    vec_to_mat(&mut v, m, Some(2_f64.sqrt()));
    assert_float_eq!(ref_m, m, abs_all <= 0.5);
    mat_to_vec(m, &mut v, Some(2_f64.sqrt()));
    assert_float_eq!(ref_v, &v, abs_all <= 1e-6);
}
