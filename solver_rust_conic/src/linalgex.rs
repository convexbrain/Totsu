// TODO: generic LinAlgEx

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
