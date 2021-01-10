use crate::solver::Operator;
use crate::linalg::{LinAlgEx, F64BLAS};

//

#[derive(Debug, Clone, Copy, PartialEq)]
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

    fn op_impl(&self, transpose: bool, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        match self.typ {
            MatType::General(nr, nc) => {
                if nr > 0 && nc > 0 {
                    F64BLAS::transform_ge(transpose, nr, nc, alpha, self.as_ref(), x, beta, y)
                }
            },
            MatType::SymPack(n) => {
                if n > 0 {
                    F64BLAS::transform_sp(n, alpha, self.as_ref(), x, beta, y)
                }
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


#[test]
fn test_matop1() {
    use float_eq::assert_float_eq;

    let array = &[ // column-major, upper-triangle (seen as if transposed)
        1.,
        2.,  3.,
        4.,  5.,  6.,
        7.,  8.,  9., 10.,
       11., 12., 13., 14., 15.,
    ];
    let ref_array = &[
        1.,  2.,  4.,  7., 11.,
        2.,  3.,  5.,  8., 12.,
        4.,  5.,  6.,  9., 13.,
        7.,  8.,  9., 10., 14.,
       11., 12., 13., 14., 15.,
    ];
    let x = &mut[0.; 5];
    let y = &mut[0.; 5];

    let m = MatOp::new(MatType::SymPack(5), array);

    for i in 0.. x.len() {
        x[i] = 1.;
        m.op(1., x, 0., y);
        assert_float_eq!(y.as_ref(), ref_array[i * 5 .. i * 5 + 5].as_ref(), abs_all <= 1e-3);
        x[i] = 0.;
    }
}
