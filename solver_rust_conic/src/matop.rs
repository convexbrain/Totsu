// TODO: no blas/lapack

use crate::solver::Operator;
use crate::solver::LinAlg;
use crate::linalg::F64BLAS;
use core::ops::{Index, IndexMut};

pub struct MatOp<'a>
{
    n_row: usize,
    n_col: usize,
    array: &'a[f64]
}

impl<'a> MatOp<'a>
{
    pub fn new((n_row, n_col): (usize, usize), array: &'a[f64]) -> Self
    {
        assert_eq!(n_row * n_col, array.len());

        MatOp {
            n_row, n_col, array,
        }
    }

    fn op_impl(&self, trans: bool, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
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
            cblas::Layout::ColumnMajor, trans,
            self.n_row as i32, self.n_col as i32,
            alpha, self.array, self.n_row as i32,
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

impl<'a> AsRef<[f64]> for MatOp<'a>
{
    fn as_ref(&self) -> &[f64]
    {
        self.array
    }
}

//

#[derive(Debug)]
pub struct MatBuilder<'a>
{
    n_row: usize,
    n_col: usize,
    array: &'a mut[f64],
}

impl<'a> MatBuilder<'a>
{
    pub fn new((n_row, n_col): (usize, usize), array: &'a mut[f64]) -> Self
    {
        assert_eq!(n_row * n_col, array.len());

        MatBuilder {
            n_row, n_col, array,
        }
    }

    // Each column is a symmetric matrix, packing the lower triangle by rows.
    pub fn build_sym(self) -> Option<Self>
    {
        let sym_n = (((8 * self.n_row + 1) as f64).sqrt() as usize - 1) / 2;

        if sym_n * (sym_n + 1) / 2 != self.n_row {
            return None;
        }

        let (_, mut ref_a) = self.array.split_at_mut(0);
        while !ref_a.is_empty() {
            for sym_r in 0.. sym_n {
                let (sym_row, spl_a) = ref_a.split_at_mut(sym_r + 1);
                ref_a = spl_a;
        
                let (sym_row_nondiag, _) = sym_row.split_at_mut(sym_r);
                F64BLAS::scale(2_f64.sqrt(), sym_row_nondiag);
            }
        }
        Some(self)
    }

    fn index(&self, (r, c): (usize, usize)) -> usize
    {
        assert!(r < self.n_row);
        assert!(c < self.n_col);

        let i = r * self.n_col + c;

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

impl<'a> AsRef<[f64]> for MatBuilder<'a>
{
    fn as_ref(&self) -> &[f64]
    {
        self.array
    }
}

impl<'a> AsMut<[f64]> for MatBuilder<'a>
{
    fn as_mut(&mut self) -> &mut[f64]
    {
        &mut self.array
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
        }
    }
}


#[test]
fn test_matop1() {
    use float_eq::assert_float_eq;

    let ref_array = &[ // column-major, upper-triangle (seen as if transposed)
        1.,
        2.*1.4,  3.,
        4.*1.4,  5.*1.4,  6.,
        7.*1.4,  8.*1.4,  9.*1.4, 10.,
       11.*1.4, 12.*1.4, 13.*1.4, 14.*1.4, 15.,
        1.,
        2.*1.4,  3.,
        4.*1.4,  5.*1.4,  6.,
        7.*1.4,  8.*1.4,  9.*1.4, 10.,
       11.*1.4, 12.*1.4, 13.*1.4, 14.*1.4, 15.,
    ];
    let array = &mut[ // column-major, upper-triangle (seen as if transposed)
        1.,
        2.,  3.,
        4.,  5.,  6.,
        7.,  8.,  9., 10.,
       11., 12., 13., 14., 15.,
        1.,
        2.,  3.,
        4.,  5.,  6.,
        7.,  8.,  9., 10.,
       11., 12., 13., 14., 15.,
    ];
    MatBuilder::new((15, 2), array).build_sym().unwrap();

    assert_float_eq!(ref_array, array, abs_all <= 0.5);
}
