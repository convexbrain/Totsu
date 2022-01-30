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
        tau[0] += 1.;
    }

    fn absadd_rows(&self, sigma: &mut[f64])
    {
        sigma[self.x_sz] += 1.;
    }
}

#[test]
fn test_trans_op()
{
    use float_eq::assert_float_eq;

    let n = 32;
    let op = ProbOpC::new(n, n);
    let sz = op.size();

    let xi = vec![1.; sz.0];

    let mut yo = vec![0.; sz.1];
    op.trans_op(1., &xi, 0., &mut yo);

    let mut yo_ref = vec![0.; sz.1];
    utils::operator_ref::trans_op::<LA, _, _>(
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
    let op = ProbOpC::new(n, n);
    let sz = op.size();

    let mut tau = vec![0.; sz.1];
    op.absadd_cols(&mut tau);

    let mut tau_ref = vec![0.; sz.1];
    utils::operator_ref::absadd_cols::<LA, _, _>(
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
    let op = ProbOpC::new(n, n);
    let sz = op.size();

    let mut sigma = vec![0.; sz.0];
    op.absadd_rows(&mut sigma);

    let mut sigma_ref = vec![0.; sz.0];
    utils::operator_ref::absadd_rows::<LA, _, _>(
        op.size(),
        |x, y| op.trans_op(1., x, 0., y),
        &mut sigma_ref
    );

    assert_float_eq!(sigma, sigma_ref, abs_all <= 1e-6);
}
