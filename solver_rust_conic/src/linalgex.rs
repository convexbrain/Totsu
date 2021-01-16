use num::Float;
use crate::solver::LinAlg;

pub trait LinAlgEx<F: Float>: LinAlg<F> + Clone
{
    // y = a*mat*x + b*y
    fn transform_ge(transpose: bool, n_row: usize, n_col: usize, alpha: F, mat: &[F], x: &[F], beta: F, y: &mut[F]);
    // y = a*mat*x + b*y
    fn transform_sp(n: usize, alpha: F, mat: &[F], x: &[F], beta: F, y: &mut[F]);
    fn proj_psd_worklen(sn: usize) -> usize;
    fn proj_psd(x: &mut[F], eps_zero: F, work: &mut[F]);
}
