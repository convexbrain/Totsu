use totsu::prelude::*;

use super::LA;

//

pub struct ProbOpC
{
    x_sz: usize,
    t_sz: usize,
}

impl ProbOpC
{
    pub fn new(width: usize, height: usize) -> Self
    {
        ProbOpC {
            x_sz: width * height,
            t_sz: (width - 2) * (height - 2),
        }
    }
}

impl Operator<f64> for ProbOpC
{
    fn size(&self) -> (usize, usize)
    {
        (self.x_sz + 1 + self.t_sz, 1)
    }

    fn op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        LA::scale(beta, y);
        y[self.x_sz] += alpha * x[0];
    }

    fn trans_op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        y[0] = alpha * x[self.x_sz] + beta * y[0];
    }

    fn absadd_cols(&self, tau: &mut[f64])
    {
        totsu::operator::reffn::absadd_cols::<LA, _, _>(
            self.size(),
            |x, y| self.op(1., x, 0., y),
            tau
        );
    }

    fn absadd_rows(&self, sigma: &mut[f64])
    {
        totsu::operator::reffn::absadd_rows::<LA, _, _>(
            self.size(),
            |x, y| self.trans_op(1., x, 0., y),
            sigma
        );
    }
}
