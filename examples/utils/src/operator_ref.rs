use num_traits::Float;
use totsu::linalg::{LinAlg, LinAlgEx};

pub fn trans_op<L, O, F>(op_size: (usize, usize), op: O, alpha: F, x: &[F], beta: F, y: &mut[F])
where L: LinAlgEx<F>, O: Fn(&[F], &mut[F]), F: Float
{
    let f0 = F::zero();
    let f1 = F::one();

    let mut col = vec![f0; op_size.0];
    let mut row = vec![f0; op_size.1];
    
    let mut y_rest = y;
    for c in 0.. op_size.1 {
        row[c] = f1;
        op(&row, &mut col);
        row[c] = f0;

        let (yc, y_lh) = y_rest.split_at_mut(1);
        y_rest = y_lh;
        L::transform_ge(true, op_size.0, 1, alpha, &col, x, beta, yc);
    }
}

pub fn absadd_cols<L, O, F>(op_size: (usize, usize), op: O, tau: &mut[F])
where L: LinAlg<F>, O: Fn(&[F], &mut[F]), F: Float
{
    let f0 = F::zero();
    let f1 = F::one();

    let mut col = vec![f0; op_size.0];
    let mut row = vec![f0; op_size.1];

    for (c, t) in tau.iter_mut().enumerate() {
        row[c] = f1;
        op(&row, &mut col);
        row[c] = f0;

        *t = L::abssum(&col, 1) + *t;
    }
}

pub fn absadd_rows<L, O, F>(op_size: (usize, usize), trans_op: O, sigma: &mut[F])
where L: LinAlg<F>, O: Fn(&[F], &mut[F]), F: Float
{
    let f0 = F::zero();
    let f1 = F::one();

    let mut col = vec![f0; op_size.0];
    let mut row = vec![f0; op_size.1];

    for (r, s) in sigma.iter_mut().enumerate() {
        col[r] = f1;
        trans_op(&col, &mut row);
        col[r] = f0;

        *s = L::abssum(&row, 1) + *s;
    }
}
