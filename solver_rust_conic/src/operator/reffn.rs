use num_traits::Float;
use crate::linalg::{LinAlg, LinAlgEx};
use std::vec;

pub fn trans_op<L, O, F>((m, n): (usize, usize), op: O, alpha: F, x: &[F], beta: F, y: &mut[F])
where L: LinAlgEx<F>, O: Fn(&[F], &mut[F]), F: Float
{
    let f0 = F::zero();
    let f1 = F::one();

    let mut row = vec![f0; n];
    let mut col = vec![f0; m];
    
    let mut y_rest = y;
    for c in 0.. n {
        row[c] = f1;
        op(&row, &mut col);
        row[c] = f0;

        let (yc, y_lh) = y_rest.split_at_mut(1);
        y_rest = y_lh;
        L::transform_ge(true, m, 1, alpha, &col, x, beta, yc);
    }
}

pub fn abssum_cols<L, O, F>((m, n): (usize, usize), op: O, beta: F, tau: &mut[F])
where L: LinAlg<F>, O: Fn(&[F], &mut[F]), F: Float
{
    let f0 = F::zero();
    let f1 = F::one();

    let mut row = vec![f0; n];
    let mut col = vec![f0; m];

    for (c, t) in tau.iter_mut().enumerate() {
        row[c] = f1;
        op(&row, &mut col);
        row[c] = f0;

        *t = L::abssum(&col) + beta * *t;
    }
}

pub fn abssum_rows<L, O, F>((m, n): (usize, usize), trans_op: O, beta: F, sigma: &mut[F])
where L: LinAlg<F>, O: Fn(&[F], &mut[F]), F: Float
{
    let f0 = F::zero();
    let f1 = F::one();

    let mut row = vec![f0; n];
    let mut col = vec![f0; m];

    for (r, s) in sigma.iter_mut().enumerate() {
        col[r] = f1;
        trans_op(&col, &mut row);
        col[r] = f0;

        *s = L::abssum(&row) + beta * *s;
    }
}
