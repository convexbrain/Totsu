use totsu::prelude::*;
use totsu::MatBuild;
use totsu_core::MatOp;
use totsu_core::solver::{Operator, LinAlg};

use super::La;

//

pub struct ProbOpB<'a>
{
    x_sz: usize,
    t_sz: usize,
    target_lx_norm1: f64,
    one: MatBuild<La>,
    xh: MatOp<'a, La>,
}

impl<'a> ProbOpB<'a>
{
    pub fn new(width: usize, height: usize, ratio: f64, vec_xh: &'a[f64]) -> Self
    {
        let target_lx_norm1 = (width * height) as f64 * ratio;
        log::info!("target_lx_norm1: {}", target_lx_norm1);

        ProbOpB {
            x_sz: width * height,
            t_sz: (width - 2) * (height - 2),
            target_lx_norm1: target_lx_norm1,
            one: MatBuild::new(MatType::General(width * height, 1))
                 .by_fn(|_, _| 1.0),
            xh: MatOp::new(MatType::General(width * height, 1), vec_xh),
        }
    }
}

impl<'a> Operator<La> for ProbOpB<'a>
{
    fn size(&self) -> (usize, usize)
    {
        (self.t_sz * 2 + 1 + self.x_sz * 2 + 1 + self.x_sz, 1)
    }

    fn op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        let (y_lp_ln, y_rest) = y.split_at_mut(self.t_sz * 2);
        let (y_l1, y_rest) = y_rest.split_at_mut(1);
        let (y_xp, y_rest) = y_rest.split_at_mut(self.x_sz);
        let (y_xn, y_rest) = y_rest.split_at_mut(self.x_sz);
        let (y_sz, y_sx) = y_rest.split_at_mut(1);

        La::scale(beta, y_lp_ln);

        y_l1[0] = alpha * self.target_lx_norm1 * x[0] + beta * y_l1[0];

        La::scale(beta, y_xp);

        self.one.as_op().op(alpha, x, beta, y_xn);

        La::scale(beta, y_sz);

        self.xh.op(-alpha, x, beta, y_sx);
    }

    fn trans_op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        let (_x_lp_ln, x_rest) = x.split_at(self.t_sz * 2);
        let (x_l1, x_rest) = x_rest.split_at(1);
        let (_x_xp, x_rest) = x_rest.split_at(self.x_sz);
        let (x_xn, x_rest) = x_rest.split_at(self.x_sz);
        let (_x_sz, x_sx) = x_rest.split_at(1);

        self.one.as_op().trans_op(alpha, x_xn, beta, y);
        self.xh.trans_op(-alpha, x_sx, 1.0, y);
        y[0] += alpha * self.target_lx_norm1 * x_l1[0];
    }

    fn absadd_cols(&self, tau: &mut[f64])
    {
        tau[0] += self.target_lx_norm1.abs() + self.x_sz as f64;
        self.xh.absadd_cols(tau);
    }

    fn absadd_rows(&self, sigma: &mut[f64])
    {
        let (_sigma_lp_ln, sigma_rest) = sigma.split_at_mut(self.t_sz * 2);
        let (sigma_l1, sigma_rest) = sigma_rest.split_at_mut(1);
        let (_sigma_xp, sigma_rest) = sigma_rest.split_at_mut(self.x_sz);
        let (sigma_xn, sigma_rest) = sigma_rest.split_at_mut(self.x_sz);
        let (_sigma_sz, sigma_sx) = sigma_rest.split_at_mut(1);

        sigma_l1[0] += self.target_lx_norm1.abs();
        La::adds(1., sigma_xn);
        self.xh.absadd_rows(sigma_sx);
    }
}

#[test]
fn test_trans_op()
{
    use float_eq::assert_float_eq;

    let n = 32;
    let vec_xh = vec![1.0; n * n];
    let op = ProbOpB::new(n, n, 1.0, &vec_xh);
    let sz = op.size();

    let xi = vec![1.; sz.0];

    let mut yo = vec![0.; sz.1];
    op.trans_op(1., &xi, 0., &mut yo);

    let mut yo_ref = vec![0.; sz.1];
    utils::operator_ref::trans_op::<La, _>(
        op.size(),
        |x, y| op.op(1., x, 0., y),
        1., &xi,
        0., &mut yo_ref);

    assert_float_eq!(yo, yo_ref, abs_all <= 1e-6);
}

#[test]
fn test_abssum_cols()
{
    use float_eq::assert_float_eq;

    let n = 32;
    let vec_xh = vec![1.0; n * n];
    let op = ProbOpB::new(n, n, 1.0, &vec_xh);
    let sz = op.size();

    let mut tau = vec![0.; sz.1];
    op.absadd_cols(&mut tau);

    let mut tau_ref = vec![0.; sz.1];
    utils::operator_ref::absadd_cols::<La, _>(
        op.size(),
        |x, y| op.op(1., x, 0., y),
        &mut tau_ref
    );

    assert_float_eq!(tau, tau_ref, abs_all <= 1e-6);
}

#[test]
fn test_abssum_rows()
{
    use float_eq::assert_float_eq;

    let n = 32;
    let vec_xh = vec![1.0; n * n];
    let op = ProbOpB::new(n, n, 1.0, &vec_xh);
    let sz = op.size();

    let mut sigma = vec![0.; sz.0];
    op.absadd_rows(&mut sigma);

    let mut sigma_ref = vec![0.; sz.0];
    utils::operator_ref::absadd_rows::<La, _>(
        op.size(),
        |x, y| op.trans_op(1., x, 0., y),
        &mut sigma_ref
    );

    assert_float_eq!(sigma, sigma_ref, abs_all <= 1e-6);
}
