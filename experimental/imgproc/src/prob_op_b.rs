use totsu::prelude::*;
use totsu::operator::MatBuild;

use super::LA;

//

pub struct ProbOpB<'a>
{
    x_sz: usize,
    t_sz: usize,
    target_lx_norm1: f64,
    one: MatBuild<LA, f64>,
    xh: MatOp<'a, LA, f64>,
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

impl<'a> Operator<f64> for ProbOpB<'a>
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

        LA::scale(beta, y_lp_ln);

        y_l1[0] = alpha * self.target_lx_norm1 * x[0] + beta * y_l1[0];

        LA::scale(beta, y_xp);

        self.one.op(alpha, x, beta, y_xn);

        LA::scale(beta, y_sz);

        self.xh.op(-alpha, x, beta, y_sx);
    }

    fn trans_op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        let (_x_lp_ln, x_rest) = x.split_at(self.t_sz * 2);
        let (x_l1, x_rest) = x_rest.split_at(1);
        let (_x_xp, x_rest) = x_rest.split_at(self.x_sz);
        let (x_xn, x_rest) = x_rest.split_at(self.x_sz);
        let (_x_sz, x_sx) = x_rest.split_at(1);

        self.one.trans_op(alpha, x_xn, beta, y);
        self.xh.trans_op(-alpha, x_sx, 1.0, y);
        y[0] += alpha * self.target_lx_norm1 * x_l1[0];
    }

    fn absadd_cols(&self, tau: &mut[f64])
    {
        utils::operator_ref::absadd_cols::<LA, _, _>(
            self.size(),
            |x, y| self.op(1., x, 0., y),
            tau
        );
    }

    fn absadd_rows(&self, sigma: &mut[f64])
    {
        utils::operator_ref::absadd_rows::<LA, _, _>(
            self.size(),
            |x, y| self.trans_op(1., x, 0., y),
            sigma
        );
    }
}
