// TODO: no blas/lapack

use crate::solver::Operator;

//

#[derive(Debug, Clone, Copy)]
pub enum MatType
{
    General(usize, usize),
    SymPack(usize),
}

impl MatType
{
    pub fn len(&self) -> usize
    {
        match self {
            MatType::General(n_row, n_col) => n_row * n_col,
            MatType::SymPack(n) => n * (n + 1) / 2,
        }
    }

    pub fn size(&self) -> (usize, usize)
    {
        match self {
            MatType::General(n_row, n_col) => (*n_row, *n_col),
            MatType::SymPack(n) => (*n, *n),
        }
    }
}

//

#[derive(Debug, Clone)]
pub struct MatOp<'a>
{
    typ: MatType,
    array: &'a[f64]
}

impl<'a> MatOp<'a>
{
    pub fn new(typ: MatType, array: &'a[f64]) -> Self
    {
        assert_eq!(typ.len(), array.len());

        MatOp {
            typ, array,
        }
    }

    fn op_impl(&self, trans: bool, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        let (nr, nc) = self.typ.size();

        let trans = if trans {
            assert_eq!(x.len(), nr);
            assert_eq!(y.len(), nc);
    
            cblas::Transpose::Ordinary
        } else {
            assert_eq!(x.len(), nc);
            assert_eq!(y.len(), nr);
    
            cblas::Transpose::None
        };

        match self.typ {
            MatType::General(_, _) => {
                unsafe { cblas::dgemv(
                    cblas::Layout::ColumnMajor, trans,
                    nr as i32, nc as i32,
                    alpha, self.array, nr as i32,
                    x, 1,
                    beta, y, 1
                ) }
            },
            MatType::SymPack(_) => {
                unsafe { cblas::dspmv(
                    cblas::Layout::ColumnMajor, cblas::Part::Upper,
                    nr as i32,
                    alpha, self.array,
                    x, 1,
                    beta, y, 1
                ) }
            },
        }
    }
}

impl<'a> Operator<f64> for MatOp<'a>
{
    fn size(&self) -> (usize, usize)
    {
        self.typ.size()
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
