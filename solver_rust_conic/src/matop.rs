// TODO: no blas/lapack

use crate::solver::Operator;

pub struct MatOp<'a>
{
    n_row: usize,
    n_col: usize,
    array: &'a[f64],
}

impl<'a> MatOp<'a>
{
    pub fn new((n_row, n_col): (usize, usize), array: &'a[f64]) -> Self
    {
        assert_eq!(n_row * n_col, array.len());

        MatOp {
            n_row, n_col, array
        }
    }
}

impl<'a> Operator for MatOp<'a>
{
    fn size(&self) -> (usize, usize)
    {
        (self.n_row, self.n_col)
    }

    fn op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        assert_eq!(x.len(), self.n_col);
        assert_eq!(y.len(), self.n_row);
        
        unsafe { cblas::dgemv(
            cblas::Layout::RowMajor, cblas::Transpose::None,
            self.n_row as i32, self.n_col as i32,
            alpha, self.array, self.n_col as i32,
            x, 1,
            beta, y, 1
        ) }
    }

    fn trans_op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        assert_eq!(x.len(), self.n_row);
        assert_eq!(y.len(), self.n_col);
        
        unsafe { cblas::dgemv(
            cblas::Layout::RowMajor, cblas::Transpose::Ordinary,
            self.n_row as i32, self.n_col as i32,
            alpha, self.array, self.n_col as i32,
            x, 1,
            beta, y, 1
        ) }
    }
}
