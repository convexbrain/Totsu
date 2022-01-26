use totsu::prelude::*;
use totsu::operator::MatBuild;

use super::LA;

use super::laplacian::Laplacian;

//

pub struct ProbOpA
{
    x_sz: usize,
    t_sz: usize,
    lap: Laplacian,
    one: MatBuild<LA, f64>,
}

impl ProbOpA
{
    pub fn new(width: usize, height: usize) -> Self
    {
        ProbOpA {
            x_sz: width * height,
            t_sz: (width - 2) * (height - 2),
            lap: Laplacian::new(width, height),
            one: MatBuild::new(MatType::General((width - 2) * (height - 2), 1))
                 .by_fn(|_, _| 1.0),
        }
    }
}

impl Operator<f64> for ProbOpA
{
    fn size(&self) -> (usize, usize)
    {
        (self.t_sz * 2 + 1 + self.x_sz * 2 + 1 + self.x_sz, self.x_sz + 1 + self.t_sz)
    }

    fn op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        let (x_x, x_rest) = x.split_at(self.x_sz);
        let (x_z, x_t) = x_rest.split_at(1);

        let (y_lp, y_rest) = y.split_at_mut(self.t_sz);
        let (y_ln, y_rest) = y_rest.split_at_mut(self.t_sz);
        let (y_l1, y_rest) = y_rest.split_at_mut(1);
        let (y_xp, y_rest) = y_rest.split_at_mut(self.x_sz);
        let (y_xn, y_rest) = y_rest.split_at_mut(self.x_sz);
        let (y_sz, y_sx) = y_rest.split_at_mut(1);

        self.lap.op(alpha, x_x, beta, y_lp);
        LA::add(-alpha, x_t, y_lp);

        self.lap.op(-alpha, x_x, beta, y_ln);
        LA::add(-alpha, x_t, y_ln);

        self.one.trans_op(alpha, x_t, beta, y_l1);

        LA::scale(beta, y_xp);
        LA::add(-alpha, x_x, y_xp);

        LA::scale(beta, y_xn);
        LA::add(alpha, x_x, y_xn);

        y_sz[0] = -alpha * x_z[0] + beta * y_sz[0];

        LA::scale(beta, y_sx);
        LA::add(-alpha, x_x, y_sx);
    }

    fn trans_op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        let (y_x, y_rest) = y.split_at_mut(self.x_sz);
        let (y_z, y_t) = y_rest.split_at_mut(1);

        let (x_lp, x_rest) = x.split_at(self.t_sz);
        let (x_ln, x_rest) = x_rest.split_at(self.t_sz);
        let (x_l1, x_rest) = x_rest.split_at(1);
        let (x_xp, x_rest) = x_rest.split_at(self.x_sz);
        let (x_xn, x_rest) = x_rest.split_at(self.x_sz);
        let (x_sz, x_sx) = x_rest.split_at(1);

        self.lap.trans_op(alpha, x_lp, beta, y_x);
        self.lap.trans_op(-alpha, x_ln, 1.0, y_x);
        LA::add(-alpha, x_xp, y_x);
        LA::add(alpha, x_xn, y_x);
        LA::add(-alpha, x_sx, y_x);

        y_z[0] = -alpha * x_sz[0] + beta * y_z[0];

        self.one.op(alpha, x_l1, beta, y_t);
        LA::add(-alpha, x_lp, y_t);
        LA::add(-alpha, x_ln, y_t);
    }

    fn absadd_cols(&self, tau: &mut[f64])
    {
        // TODO
        utils::operator_ref::absadd_cols::<LA, _, _>(
            self.size(),
            |x, y| self.op(1., x, 0., y),
            tau
        );
    }

    fn absadd_rows(&self, sigma: &mut[f64])
    {
        // TODO
        utils::operator_ref::absadd_rows::<LA, _, _>(
            self.size(),
            |x, y| self.trans_op(1., x, 0., y),
            sigma
        );
    }
}

#[test]
fn test_trans_op()
{
    use float_eq::assert_float_eq;

    let n = 32;
    let op = ProbOpA::new(n, n);
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
    let op = ProbOpA::new(n, n);
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
    let op = ProbOpA::new(n, n);
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
