use num::Float;

pub trait LinAlg<F: Float>
{
    fn norm(x: &[F]) -> F;
    fn copy(x: &[F], y: &mut[F]);
    fn scale(alpha: F, x: &mut[F]);
    // y = a*x + y
    fn add(alpha: F, x: &[F], y: &mut[F]);
    fn abssum(x: &[F]) -> F;
    // y = a*mat*x + b*y
    fn transform_di(alpha: F, mat: &[F], x: &[F], beta: F, y: &mut[F]);
}

pub trait LinAlgEx<F: Float>: LinAlg<F> + Clone
{
    // y = a*mat*x + b*y
    fn transform_ge(transpose: bool, n_row: usize, n_col: usize, alpha: F, mat: &[F], x: &[F], beta: F, y: &mut[F]);
    // y = a*mat*x + b*y
    fn transform_sp(n: usize, alpha: F, mat: &[F], x: &[F], beta: F, y: &mut[F]);
    fn proj_psd_worklen(sn: usize) -> usize;
    fn proj_psd(x: &mut[F], eps_zero: F, work: &mut[F]);
    fn sqrt_spmat_worklen(n: usize) -> usize;
    fn sqrt_spmat(x: &mut[F], eps_zero: F, work: &mut[F]);
}

//

mod floatgeneric; // core, Float
mod f64lapack;    // core, f64(cblas/lapacke)

pub use floatgeneric::*;
pub use f64lapack::*;
