use num_traits::Float;
use core::marker::PhantomData;
use crate::linalg::LinAlgEx;
use super::Operator;

//

/// Matrix type and size
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatType
{
    /// General matrix with a number of rows and a number of columns.
    General(usize, usize),
    /// Symmetric matrix, supplied in packed form, with a number of rows and columns.
    SymPack(usize),
}

impl MatType
{
    /// Length of array to store a [`MatType`] matrix.
    /// 
    /// Returns the length.
    pub fn len(&self) -> usize
    {
        match self {
            MatType::General(n_row, n_col) => n_row * n_col,
            MatType::SymPack(n) => n * (n + 1) / 2,
        }
    }

    /// Size of a [`MatType`] matrix.
    /// 
    /// Returns a tuple of a number of rows and a number of columns.
    pub fn size(&self) -> (usize, usize)
    {
        match self {
            MatType::General(n_row, n_col) => (*n_row, *n_col),
            MatType::SymPack(n) => (*n, *n),
        }
    }
}

//

/// Matrix operator
/// 
/// Matrix struct which borrows a slice of data array and implements [`Operator`].
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
    /// Creates an instance
    /// 
    /// Returns [`MatOp`] instance.
    /// * `typ`: Matrix type and size.
    /// * `array`: data array slice.
    ///   Column-major matrix data shall be stored if [`MatType::General`].
    ///   Symmetric packed form (the upper-triangular part in column-wise) of matrix data shall be stored if [`MatType::SymPack`].
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
                else {
                    L::scale(beta, y);
                }
            },
            MatType::SymPack(n) => {
                if n > 0 {
                    L::transform_sp(n, alpha, self.as_ref(), x, beta, y)
                }
                else {
                    L::scale(beta, y);
                }
            },
        }
    }

    fn absadd_impl(&self, colwise: bool, y: &mut[F])
    {
        match self.typ {
            MatType::General(nr, nc) => {
                if colwise {
                    assert_eq!(nc, y.len());
                
                    let mut array = self.array;
                    for e in y {
                        let (col, rest) = array.split_at(nr);
                        array = rest;
                        *e = L::abssum(col, 1) + *e;
                    }
                }
                else {
                    assert_eq!(nr, y.len());
                
                    let mut array = self.array;
                    for e in y {
                        *e = L::abssum(array, nr) + *e;
                        let (_, rest) = array.split_at(1);
                        array = rest;
                    }
                }
            },
            MatType::SymPack(n) => {
                assert_eq!(n, y.len());

                let mut array = self.array;
                for n in 0.. {
                    let (col, rest) = array.split_at(n + 1);
                    y[n] = L::abssum(col, 1) + y[n];
                    for i in 0.. n {
                        y[i] = y[i] + col[i].abs();
                    }
                    array = rest;
                    if array.len() == 0 {
                        break;
                    }
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

    fn absadd_cols(&self, tau: &mut[F])
    {
        self.absadd_impl(true, tau);
    }

    fn absadd_rows(&self, sigma: &mut[F])
    {
        self.absadd_impl(false, sigma);
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
fn test_matop1()
{
    use float_eq::assert_float_eq;
    use crate::linalg::FloatGeneric;

    type L = FloatGeneric<f64>;

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

    let m = MatOp::<L, _>::new(MatType::SymPack(5), array);

    for i in 0.. x.len() {
        x[i] = 1.;
        m.op(1., x, 0., y);
        assert_float_eq!(y.as_ref(), ref_array[i * 5 .. i * 5 + 5].as_ref(), abs_all <= 1e-3);
        x[i] = 0.;
    }
}
