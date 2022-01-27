use totsu::prelude::*;

use super::LA;

//

fn adds(s: f64, y: &mut[f64], incy: usize)
{
    let one = &[1.];

    unsafe { cblas::daxpy(((y.len() + (incy - 1)) / incy) as i32, s, one, 0, y, incy as i32) }
}

//

pub struct Laplacian
{
    w: usize,
    h: usize,
    c0: f64,
    c1: f64,
    c2: f64,
}

impl Laplacian
{
    pub fn new(w: usize, h: usize) -> Self
    {
        assert!(w >= 5);
        assert!(h >= 5);

        Laplacian {
            w,
            h,
            c0: 1.0 - 4.0 / 16.0,
            c1: -2.0 / 16.0,
            c2: -1.0/ 16.0,
        }
    }

    fn pos(&self, x: isize, y: isize) -> usize
    {
        let x = if x < 0 {self.w as isize + x} else {x} as usize;
        let y = if y < 0 {self.h as isize + y} else {y} as usize;

        assert!(x < self.w);
        assert!(y < self.h);

        x + y * self.w
    }

    pub fn absadd_cols_alpha(&self, alpha: f64, tau: &mut[f64])
    {
        /*
           0                  w
            [0][1][2]... [1][0]
            [1][3][4]... [3][1]
            [2][4][5]... [4][2]
                     ...

            [1][3][4]... [3][1]
            [0][1][2]... [1][0]
           h
        */

        let pos = |x, y| self.pos(x, y);

        // [0]
        for i in [pos(0,  0), pos(-1,  0),
                  pos(0, -1), pos(-1, -1)] {
            let a = self.c2.abs();
            tau[i] += alpha * a;
        }
        // [1]
        for i in [pos(1,  0), pos(-2,  0),
                  pos(0,  1), pos(-1,  1),
                  pos(0, -2), pos(-1, -2),
                  pos(1, -1), pos(-2, -1)] {
            let a = self.c1.abs() + self.c2.abs();
            tau[i] += alpha * a;
        }
        // [3]
        for i in [pos(1,  1), pos(-2,  1),
                  pos(1, -2), pos(-2, -2)] {
            let a = self.c0.abs() + self.c1.abs() * 2. + self.c2.abs();
            tau[i] += alpha * a;
        }

        // [2]
        let a = self.c1.abs() + self.c2.abs() * 2.;
        LA::adds(alpha * a, tau.split_at_mut(pos(2,  0)).1.split_at_mut(self.w - 4).0);
        LA::adds(alpha * a, tau.split_at_mut(pos(2, -1)).1.split_at_mut(self.w - 4).0);
        adds(alpha * a, tau.split_at_mut(pos( 0, 2)).1.split_at_mut(self.w * (self.h - 4)).0, self.w);
        adds(alpha * a, tau.split_at_mut(pos(-1, 2)).1.split_at_mut(self.w * (self.h - 4)).0, self.w);

        // [4]
        let a = self.c0.abs() + self.c1.abs() * 3. + self.c2.abs() * 2.;
        LA::adds(alpha * a, tau.split_at_mut(pos(2,  1)).1.split_at_mut(self.w - 4).0);
        LA::adds(alpha * a, tau.split_at_mut(pos(2, -2)).1.split_at_mut(self.w - 4).0);
        adds(alpha * a, tau.split_at_mut(pos( 1, 2)).1.split_at_mut(self.w * (self.h - 4)).0, self.w);
        adds(alpha * a, tau.split_at_mut(pos(-2, 2)).1.split_at_mut(self.w * (self.h - 4)).0, self.w);

        // [5]
        let a = self.c0.abs() + self.c1.abs() * 4. + self.c2.abs() * 4.;
        for i in 2.. (self.h - 2) {
            let i = i as isize;
            LA::adds(alpha * a, tau.split_at_mut(pos(2, i)).1.split_at_mut(self.w - 4).0);
        }
    }
}

impl Operator<f64> for Laplacian
{
    fn size(&self) -> (usize, usize)
    {
        ((self.w - 2) * (self.h - 2), self.w * self.h)
    }

    fn op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        LA::scale(beta, y);

        let mut y_rest = y;
        for cy in 0..(self.h - 2) {
            let (y_line, y_lh) = y_rest.split_at_mut(self.w - 2);
            y_rest = y_lh;

            let x_0 = x.split_at(cy * self.w).1;
            let x_00 = x_0.split_at(self.w - 2).0;
            let x_01 = x_0.split_at(1).1.split_at(self.w - 2).0;
            let x_02 = x_0.split_at(2).1.split_at(self.w - 2).0;
            LA::add(alpha * self.c2, x_00, y_line);
            LA::add(alpha * self.c1, x_01, y_line);
            LA::add(alpha * self.c2, x_02, y_line);

            let x_1 = x.split_at((cy + 1) * self.w).1;
            let x_10 = x_1.split_at(self.w - 2).0;
            let x_11 = x_1.split_at(1).1.split_at(self.w - 2).0;
            let x_12 = x_1.split_at(2).1.split_at(self.w - 2).0;
            LA::add(alpha * self.c1, x_10, y_line);
            LA::add(alpha * self.c0, x_11, y_line);
            LA::add(alpha * self.c1, x_12, y_line);

            let x_2 = x.split_at((cy + 2) * self.w).1;
            let x_20 = x_2.split_at(self.w - 2).0;
            let x_21 = x_2.split_at(1).1.split_at(self.w - 2).0;
            let x_22 = x_2.split_at(2).1.split_at(self.w - 2).0;
            LA::add(alpha * self.c2, x_20, y_line);
            LA::add(alpha * self.c1, x_21, y_line);
            LA::add(alpha * self.c2, x_22, y_line);
        }
    }

    fn trans_op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        LA::scale(beta, y);
        
        let mut x_rest = x;
        for cy in 0..(self.h - 2) {
            let (x_line, x_lh) = x_rest.split_at(self.w - 2);
            x_rest = x_lh;

            let y_0 = y.split_at_mut(cy * self.w).1;
            let y_00 = y_0.split_at_mut(self.w - 2).0;
            LA::add(alpha * self.c2, x_line, y_00);
            let y_01 = y_0.split_at_mut(1).1.split_at_mut(self.w - 2).0;
            LA::add(alpha * self.c1, x_line, y_01);
            let y_02 = y_0.split_at_mut(2).1.split_at_mut(self.w - 2).0;
            LA::add(alpha * self.c2, x_line, y_02);

            let y_1 = y.split_at_mut((cy + 1) * self.w).1;
            let y_10 = y_1.split_at_mut(self.w - 2).0;
            LA::add(alpha * self.c1, x_line, y_10);
            let y_11 = y_1.split_at_mut(1).1.split_at_mut(self.w - 2).0;
            LA::add(alpha * self.c0, x_line, y_11);
            let y_12 = y_1.split_at_mut(2).1.split_at_mut(self.w - 2).0;
            LA::add(alpha * self.c1, x_line, y_12);

            let y_2 = y.split_at_mut((cy + 2) * self.w).1;
            let y_20 = y_2.split_at_mut(self.w - 2).0;
            LA::add(alpha * self.c2, x_line, y_20);
            let y_21 = y_2.split_at_mut(1).1.split_at_mut(self.w - 2).0;
            LA::add(alpha * self.c1, x_line, y_21);
            let y_22 = y_2.split_at_mut(2).1.split_at_mut(self.w - 2).0;
            LA::add(alpha * self.c2, x_line, y_22);
        }
    }

    fn absadd_cols(&self, tau: &mut[f64])
    {
        self.absadd_cols_alpha(1., tau);
    }

    fn absadd_rows(&self, sigma: &mut[f64])
    {
        let a = self.c0.abs() + self.c1.abs() * 4. + self.c2.abs() * 4.;
        LA::adds(a, sigma);
    }
}

#[test]
fn test_trans_op()
{
    use float_eq::assert_float_eq;

    let n = 32;
    let op = Laplacian::new(n, n);
    let sz = op.size();

    let xi = vec![1.; sz.0];

    let mut yo = vec![0.; sz.1];
    op.trans_op(1., &xi, 0., &mut yo);

    let mut yo_ref = vec![0.; sz.1];
    utils::operator_ref::trans_op::<LA, _, _>(
        op.size(),
        |x, y| op.op(1., x, 0., y),
        1., &xi,
        0., &mut yo_ref
    );

    assert_float_eq!(yo, yo_ref, abs_all <= 1e-6);
}

#[test]
fn test_abssum_cols()
{
    use float_eq::assert_float_eq;

    let n = 32;
    let op = Laplacian::new(n, n);
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
    let op = Laplacian::new(n, n);
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
