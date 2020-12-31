// TODO: no blas/lapack

use crate::solver::Operator;
use crate::linalg::scale;

pub struct MatOp<'a>
{
    n_row: usize,
    n_col: usize,
    array: &'a[f64],
    col_major: bool,
}

fn sym_scale(n_row: usize, array: &mut[f64])
{
    let n = (((8 * n_row + 1) as f64).sqrt() as usize - 1) / 2;

    let mut ref_a = array;
    for d in 0.. n {
        let (r, spl_a) = ref_a.split_at_mut(n - d);
        ref_a = spl_a;

        let (_, rt) = r.split_at_mut(1);
        scale(2_f64.sqrt(), rt);
    }
}

impl<'a> MatOp<'a>
{
    // From a row-major array
    pub fn new((n_row, n_col): (usize, usize), array: &'a[f64]) -> Self
    {
        assert_eq!(n_row * n_col, array.len());

        MatOp {
            n_row, n_col, array, col_major: false
        }
    }

    // From a column-major array, each column of which is
    // upper triangular elements of symmetric matrix vectorized in row-wise
    pub fn new_sym((n_row, n_col): (usize, usize), array: &'a mut[f64]) -> Self
    {
        assert_eq!(n_row * n_col, array.len());

        sym_scale(n_row, array);

        MatOp {
            n_row, n_col, array, col_major: true
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

impl<'a> Operator for MatOp<'a>
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
