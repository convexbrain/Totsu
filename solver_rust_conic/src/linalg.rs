// TODO: no blas/lapack

pub fn norm(x: &[f64]) -> f64
{
    unsafe { cblas::dnrm2(x.len() as i32, x, 1) }
}

pub fn inner_prod(x: &[f64], y: &[f64]) -> f64
{
    assert_eq!(x.len(), y.len());

    unsafe { cblas::ddot(x.len() as i32, x, 1, y, 1) }
}

pub fn copy(x: &[f64], y: &mut[f64])
{
    assert_eq!(x.len(), y.len());

    unsafe { cblas::dcopy(x.len() as i32, x, 1, y, 1) }
}

pub fn scale(alpha: f64, x: &mut[f64])
{
    unsafe { cblas::dscal(x.len() as i32, alpha, x, 1) }
}

pub fn add(alpha: f64, x: &[f64], y: &mut[f64])
{
    assert_eq!(x.len(), y.len());

    unsafe { cblas::daxpy(x.len() as i32, alpha, x, 1, y, 1) }
}
