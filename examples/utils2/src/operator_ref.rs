use num_traits::{Zero, One};
use totsu_core::solver::{LinAlg, SliceLike};
use totsu_core::{LinAlgEx, splitm_mut};

pub fn trans_op<L, O>(op_size: (usize, usize), op: O, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
where L: LinAlgEx, O: Fn(&L::Sl, &mut L::Sl)
{
    let f0 = L::F::zero();
    let f1 = L::F::one();
    let (m, n) = op_size;
    
    let mut col_v = std::vec![f0; m];
    let mut row_v = std::vec![f0; n];
    let mut col = L::Sl::new_mut(&mut col_v);
    let mut row = L::Sl::new_mut(&mut row_v);
    
    for c in 0.. n {
        row.set(c, f1);
        op(&row, &mut col);
        row.set(c, f0);
    
        splitm_mut!(y, (_y_done; c), (yc; 1));
        L::transform_ge(true, m, 1, alpha, &col, x, beta, &mut yc);
    }
}

pub fn absadd_cols<L, O>(op_size: (usize, usize), op: O, tau: &mut L::Sl)
where L: LinAlg, O: Fn(&L::Sl, &mut L::Sl)
{
    let f0 = L::F::zero();
    let f1 = L::F::one();
    let (m, n) = op_size;
    
    let mut col_v = std::vec![f0; m];
    let mut row_v = std::vec![f0; n];
    let mut col = L::Sl::new_mut(&mut col_v);
    let mut row = L::Sl::new_mut(&mut row_v);
    
    for c in 0.. tau.len() {
        row.set(c, f1);
        op(&row, &mut col);
        row.set(c, f0);
    
        let val_tau = tau.get(c) + L::abssum(&col, 1);
        tau.set(c, val_tau);
    }
}

pub fn absadd_rows<L, O>(op_size: (usize, usize), trans_op: O, sigma: &mut L::Sl)
where L: LinAlg, O: Fn(&L::Sl, &mut L::Sl)
{
    let f0 = L::F::zero();
    let f1 = L::F::one();
    let (m, n) = op_size;
    
    let mut col_v = std::vec![f0; m];
    let mut row_v = std::vec![f0; n];
    let mut col = L::Sl::new_mut(&mut col_v);
    let mut row = L::Sl::new_mut(&mut row_v);
    
    for r in 0.. sigma.len() {
        col.set(r, f1);
        trans_op(&col, &mut row);
        col.set(r, f0);
    
        let val_sigma = sigma.get(r) + L::abssum(&row, 1);
        sigma.set(r, val_sigma);
    }
}
