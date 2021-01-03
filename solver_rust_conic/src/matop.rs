// TODO: no blas/lapack

use crate::solver::Operator;
use crate::solver::LinAlg;
use crate::linalg::F64BLAS;
use core::ops::{Index, IndexMut};

pub struct MatOp<'a>
{
    n_row: usize,
    n_col: usize,
    array: &'a[f64],
    col_major: bool,
}

impl<'a> MatOp<'a>
{
    pub fn new((n_row, n_col): (usize, usize), col_major: bool, array: &'a[f64]) -> Self
    {
        assert_eq!(n_row * n_col, array.len());

        MatOp {
            n_row, n_col, array, col_major
        }
    }

    fn op_impl(&self, trans: bool, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        let (layout, lda) = if self.col_major {
            (cblas::Layout::ColumnMajor, self.n_row)
        } else {
            (cblas::Layout::RowMajor, self.n_col)
        };
        
        let trans = if trans {
            assert_eq!(x.len(), self.n_row);
            assert_eq!(y.len(), self.n_col);
    
            cblas::Transpose::Ordinary
        } else {
            assert_eq!(x.len(), self.n_col);
            assert_eq!(y.len(), self.n_row);
    
            cblas::Transpose::None
        };
        
        unsafe { cblas::dgemv(
            layout, trans,
            self.n_row as i32, self.n_col as i32,
            alpha, self.array, lda as i32,
            x, 1,
            beta, y, 1
        ) }
    }
}

impl<'a> Operator<f64> for MatOp<'a>
{
    fn size(&self) -> (usize, usize)
    {
        (self.n_row, self.n_col)
    }

    fn op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        self.op_impl(false, alpha, x, beta, y);
    }

    fn trans_op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        self.op_impl(true, alpha, x, beta, y);
    }
}

//

pub struct MatBuilder<'a>
{
    n_row: usize,
    n_col: usize,
    array: &'a mut[f64],
    col_major: bool,
}

impl<'a> MatBuilder<'a>
{
    pub fn new((n_row, n_col): (usize, usize), col_major: bool, array: &'a mut[f64]) -> Self
    {
        assert_eq!(n_row * n_col, array.len());

        MatBuilder {
            n_row, n_col, array, col_major
        }
    }

    // Each column is upper triangular elements of symmetric matrix vectorized in row-wise
    pub fn build_sym(self) -> Option<Self>
    {
        let n = (((8 * self.n_row + 1) as f64).sqrt() as usize - 1) / 2;

        if n * (n + 1) / 2 != self.n_row {
            return None;
        }

        let (_, mut ref_a) = self.array.split_at_mut(0);
        if self.col_major {
            while !ref_a.is_empty() {
                for sym_r in 0.. n {
                    let (sym_row, spl_a) = ref_a.split_at_mut(n - sym_r);
                    ref_a = spl_a;
            
                    let (_, sym_row_ndiag) = sym_row.split_at_mut(1);
                    F64BLAS::scale(2_f64.sqrt(), sym_row_ndiag);
                }
            }
        }
        else {
            for sym_r in 0.. n {
                let (_, spl_a) = ref_a.split_at_mut(self.n_col);
                ref_a = spl_a;

                let (sym_row, spl_a) = ref_a.split_at_mut(self.n_col * (n - 1 - sym_r));
                ref_a = spl_a;
                F64BLAS::scale(2_f64.sqrt(), sym_row);
            }
        }
        Some(self)
    }

    fn index(&self, (r, c): (usize, usize)) -> usize
    {
        assert!(r < self.n_row);
        assert!(c < self.n_col);

        let i = if self.col_major {
            r * self.n_row + c
        }
        else {
            c * self.n_col + r
        };

        assert!(i < self.array.len());

        i
    }
}

impl<'a> Index<(usize, usize)> for MatBuilder<'a>
{
    type Output = f64;
    fn index(&self, index: (usize, usize)) -> &f64
    {
        let i = self.index(index);

        &self.array[i]
    }
}

impl<'a> IndexMut<(usize, usize)> for MatBuilder<'a>
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut f64
    {
        let i = self.index(index);

        &mut self.array[i]
    }
}

impl<'a> From<MatBuilder<'a>> for MatOp<'a>
{
    fn from(m: MatBuilder<'a>) -> Self
    {
        MatOp {
            n_row: m.n_row,
            n_col: m.n_col,
            array: m.array,
            col_major: m.col_major,
        }
    }
}
