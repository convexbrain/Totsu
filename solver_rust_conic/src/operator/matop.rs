use num::Float;
use core::marker::PhantomData;
use crate::linalg::LinAlgEx;
use super::Operator;

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
pub struct MatOp<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    ph_l: PhantomData<L>,
    typ: MatType,
    array: &'a[F]
}

impl<'a, L, F> MatOp<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    pub fn new(typ: MatType, array: &'a[F]) -> Self
    {
        assert_eq!(typ.len(), array.len());

        MatOp {
            ph_l: PhantomData,
            typ, array,
        }
    }

    fn op_impl(&self, transpose: bool, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        match self.typ {
            MatType::General(nr, nc) => {
                if nr > 0 && nc > 0 {
                    L::transform_ge(transpose, nr, nc, alpha, self.as_ref(), x, beta, y)
                }
            },
            MatType::SymPack(n) => {
                if n > 0 {
                    L::transform_sp(n, alpha, self.as_ref(), x, beta, y)
                }
            },
        }
    }
}

impl<'a, L, F> Operator<F> for MatOp<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn size(&self) -> (usize, usize)
    {
        self.typ.size()
    }

    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        self.op_impl(false, alpha, x, beta, y);
    }

    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        self.op_impl(true, alpha, x, beta, y);
    }
}

impl<'a, L, F> AsRef<[F]> for MatOp<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn as_ref(&self) -> &[F]
    {
        self.array
    }
}

//

#[test]
fn test_matop1() {
    use float_eq::assert_float_eq;
    use crate::linalg::F64LAPACK;

    type AMatOp<'a> = MatOp<'a, F64LAPACK, f64>;

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

    let m = AMatOp::new(MatType::SymPack(5), array);

    for i in 0.. x.len() {
        x[i] = 1.;
        m.op(1., x, 0., y);
        assert_float_eq!(y.as_ref(), ref_array[i * 5 .. i * 5 + 5].as_ref(), abs_all <= 1e-3);
        x[i] = 0.;
    }
}
