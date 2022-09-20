use num_traits::Float;
use crate::solver::{SliceLike, SliceRef, Operator};
use crate::{LinAlgEx, splitm};

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
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
/// 
/// Matrix struct which borrows a slice of data array and implements [`Operator`].
#[derive(Debug)]
pub struct MatOp<'a, L: LinAlgEx>
{
    typ: MatType,
    array: SliceRef<'a, L::Sl>
}

impl<'a, L: LinAlgEx> MatOp<'a, L>
{
    /// Creates an instance
    /// 
    /// Returns [`MatOp`] instance.
    /// * `typ`: Matrix type and size.
    /// * `array`: data array slice.
    ///   Column-major matrix data shall be stored if [`MatType::General`].
    ///   Symmetric packed form (the upper-triangular part in column-wise) of matrix data shall be stored if [`MatType::SymPack`].
    pub fn new(typ: MatType, array: &'a[L::F]) -> Self
    {
        assert_eq!(typ.len(), array.len());

        MatOp {
            typ,
            array: L::Sl::new_ref(array)
        }
    }

    fn op_impl(&self, transpose: bool, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        match self.typ {
            MatType::General(nr, nc) => {
                if nr > 0 && nc > 0 {
                    L::transform_ge(transpose, nr, nc, alpha, &self.array, x, beta, y)
                }
                else {
                    L::scale(beta, y);
                }
            },
            MatType::SymPack(n) => {
                if n > 0 {
                    L::transform_sp(n, alpha, &self.array, x, beta, y)
                }
                else {
                    L::scale(beta, y);
                }
            },
        }
    }

    fn absadd_impl(&self, colwise: bool, y: &mut L::Sl)
    {
        match self.typ {
            MatType::General(nr, nc) => {
                if colwise {
                    assert_eq!(nc, y.len());
                
                    for (i, e) in y.get_mut().iter_mut().enumerate() {
                        splitm!(self.array, (_t; i * nr), (col; nr));
                        *e = L::abssum(&col, 1) + *e;
                    }
                }
                else {
                    assert_eq!(nr, y.len());
                
                    for (i, e) in y.get_mut().iter_mut().enumerate() {
                        splitm!(self.array, (_t; i), (row; nr * nc - i));
                        *e = L::abssum(&row, nr) + *e;
                    }
                }
            },
            MatType::SymPack(n) => {
                assert_eq!(n, y.len());

                let y_mut = y.get_mut();
                let mut sum = 0;
                for n in 0.. {
                    splitm!(self.array, (_t; sum), (col; n + 1));
                    sum += n + 1;
                    y_mut[n] = L::abssum(&col, 1) + y_mut[n];
                    let col_ref = col.get_ref();
                    for i in 0.. n {
                        y_mut[i] = y_mut[i] + col_ref[i].abs();
                    }
                    if sum == self.array.len() {
                        break;
                    }
                }
            },
        }
    }
}

impl<'a, L: LinAlgEx> Operator<L> for MatOp<'a, L>
{
    fn size(&self) -> (usize, usize)
    {
        self.typ.size()
    }

    fn op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        self.op_impl(false, alpha, x, beta, y);
    }

    fn trans_op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        self.op_impl(true, alpha, x, beta, y);
    }

    fn absadd_cols(&self, tau: &mut L::Sl)
    {
        self.absadd_impl(true, tau);
    }

    fn absadd_rows(&self, sigma: &mut L::Sl)
    {
        self.absadd_impl(false, sigma);
    }
}

impl<'a, L: LinAlgEx> AsRef<[L::F]> for MatOp<'a, L>
{
    fn as_ref(&self) -> &[L::F]
    {
        self.array.get_ref()
    }
}

//

#[test]
fn test_matop1()
{
    use float_eq::assert_float_eq;
    use crate::FloatGeneric;

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

    let m = MatOp::<L>::new(MatType::SymPack(5), array);

    for i in 0.. x.len() {
        x[i] = 1.;
        m.op(1., x, 0., y);
        assert_float_eq!(y.as_ref(), ref_array[i * 5 .. i * 5 + 5].as_ref(), abs_all <= 1e-3);
        x[i] = 0.;
    }
}
