use crate::matop::{MatType, MatOp};
use crate::linalgex::LinAlgEx;
use crate::solver::Operator;
use core::ops::{Index, IndexMut, Deref};
use core::marker::PhantomData;
use num::Float;

//

#[derive(Debug, Clone)]
pub struct MatBuild<L, F>
where L: LinAlgEx<F>, F: Float
{
    ph_l: PhantomData<L>,
    typ: MatType,
    array: Vec<F>,
}

impl<L, F> MatBuild<L, F>
where L: LinAlgEx<F>, F: Float
{
    pub fn new(typ: MatType) -> Self
    {
        MatBuild {
            ph_l: PhantomData,
            typ,
            array: vec![F::zero(); typ.len()],
        }
    }

    pub fn is_sympack(&self) -> bool
    {
        if let MatType::SymPack(_) = self.typ {
            true
        }
        else {
            false
        }
    }

    pub fn set_by_fn<M>(&mut self, mut func: M)
    where M: FnMut(usize, usize) -> F
    {
        match self.typ {
            MatType::General(nr, nc) => {
                for c in 0.. nc {
                    for r in 0.. nr {
                        self[(r, c)] = func(r, c);
                    }
                }
            },
            MatType::SymPack(n) => {
                for c in 0.. n {
                    for r in 0..= c {
                        self[(r, c)] = func(r, c);
                    }
                }
            },
        };
    }
    pub fn by_fn<M>(mut self, func: M) -> Self
    where M: FnMut(usize, usize) -> F
    {
        self.set_by_fn(func);
        self
    }

    pub fn set_iter_colmaj<T, I>(&mut self, iter: T)
    where T: IntoIterator<Item=I>, I: Deref<Target=F>
    {
        let mut i = iter.into_iter();
        let (nr, nc) = self.typ.size();

        for c in 0.. nc {
            for r in 0.. nr {
                if let Some(v) = i.next() {
                    self[(r, c)] = *v;
                }
                else {
                    break;
                }
            }
        }
    }
    pub fn iter_colmaj<T, I>(mut self, iter: T) -> Self
    where T: IntoIterator<Item=I>, I: Deref<Target=F>
    {
        self.set_iter_colmaj(iter);
        self
    }

    pub fn set_iter_rowmaj<T, I>(&mut self, iter: T)
    where T: IntoIterator<Item=I>, I: Deref<Target=F>
    {
        let mut i = iter.into_iter();
        let (nr, nc) = self.typ.size();

        for r in 0.. nr {
            for c in 0.. nc {
                if let Some(v) = i.next() {
                    self[(r, c)] = *v;
                }
                else {
                    break;
                }
            }
        }
    }
    pub fn iter_rowmaj<T, I>(mut self, iter: T) -> Self
    where T: IntoIterator<Item=I>, I: Deref<Target=F>
    {
        self.set_iter_rowmaj(iter);
        self
    }

    pub fn set_scale(&mut self, alpha: F)
    {
        L::scale(alpha, self.as_mut());
    }
    pub fn scale(mut self, alpha: F) -> Self
    {
        self.set_scale(alpha);
        self
    }

    pub fn set_scale_nondiag(&mut self, alpha: F)
    {
        match self.typ {
            MatType::General(nr, nc) => {
                let n = nr.min(nc);
                for c in 0.. n - 1 {
                    let i = self.index((c, c));
                    let (_, spl) = self.as_mut().split_at_mut(i + 1);
                    let (spl, _) = spl.split_at_mut(nc);
                    L::scale(alpha, spl);
                }
                let i = self.index((n, n));
                let (_, spl) = self.as_mut().split_at_mut(i + 1);
                L::scale(alpha, spl);
            },
            MatType::SymPack(n) => {
                for c in 0.. n - 1 {
                    let i = self.index((c, c));
                    let ii = self.index((c + 1, c + 1));
                    let (_, spl) = self.as_mut().split_at_mut(i + 1);
                    let (spl, _) = spl.split_at_mut(ii - i - 1);
                    L::scale(alpha, spl);
                }
            },
        }
    }
    pub fn scale_nondiag(mut self, alpha: F) -> Self
    {
        self.set_scale_nondiag(alpha);
        self
    }

    pub fn set_reshape_colvec(&mut self)
    {
        let sz = self.as_ref().len();
        self.typ = MatType::General(sz, 1);
    }
    pub fn reshape_colvec(mut self) -> Self
    {
        self.set_reshape_colvec();
        self
    }

    fn index(&self, (r, c): (usize, usize)) -> usize
    {
        let i = match self.typ {
            MatType::General(nr, nc) => {
                assert!(r < nr);
                assert!(c < nc);
                c * nr + r
            },
            MatType::SymPack(n) => {
                assert!(r < n);
                assert!(c < n);
                let (r, c) = if r <= c {
                    (r, c)
                }
                else {
                    (c, r)
                };
                c * (c + 1) / 2 + r
            },
        };

        assert!(i < self.array.len());
        i
    }
}

impl<L, F> Index<(usize, usize)> for MatBuild<L, F>
where L: LinAlgEx<F>, F: Float
{
    type Output = F;
    fn index(&self, index: (usize, usize)) -> &F
    {
        let i = self.index(index);

        &self.array[i]
    }
}

impl<L, F> IndexMut<(usize, usize)> for MatBuild<L, F>
where L: LinAlgEx<F>, F: Float
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut F
    {
        let i = self.index(index);

        &mut self.array[i]
    }
}

impl<L, F> AsRef<[F]> for MatBuild<L, F>
where L: LinAlgEx<F>, F: Float
{
    fn as_ref(&self) -> &[F]
    {
        &self.array
    }
}

impl<L, F> AsMut<[F]> for MatBuild<L, F>
where L: LinAlgEx<F>, F: Float
{
    fn as_mut(&mut self) -> &mut[F]
    {
        &mut self.array
    }
}

impl<'a, L, F> From<&'a MatBuild<L, F>> for MatOp<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn from(m: &'a MatBuild<L, F>) -> Self
    {
        MatOp::new(m.typ, m.as_ref())
    }
}

impl<L, F> Operator<F> for MatBuild<L, F>
where L: LinAlgEx<F>, F: Float
{
    fn size(&self) -> (usize, usize)
    {
        self.typ.size()
    }

    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        MatOp::from(self).op(alpha, x, beta, y);
    }

    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        MatOp::from(self).trans_op(alpha, x, beta, y);
    }
}

#[test]
fn test_matbuild1() {
    use float_eq::assert_float_eq;
    use crate::f64lapack::F64LAPACK;

    type AMatBuild = MatBuild<F64LAPACK, f64>;

    let ref_array = &[ // column-major, upper-triangle (seen as if transposed)
        1.,
        2.*1.4,  3.,
        4.*1.4,  5.*1.4,  6.,
        7.*1.4,  8.*1.4,  9.*1.4, 10.,
       11.*1.4, 12.*1.4, 13.*1.4, 14.*1.4, 15.,
    ];
    let array = &[ // column-major, upper-triangle (seen as if transposed)
        1.,  0.,  0.,  0.,  0.,
        2.,  3.,  0.,  0.,  0.,
        4.,  5.,  6.,  0.,  0.,
        7.,  8.,  9., 10.,  0.,
       11., 12., 13., 14., 15.,
    ];

    let m = AMatBuild::new(MatType::SymPack(5))
            .iter_colmaj(array)
            .scale_nondiag(1.4);

    assert_float_eq!(m.as_ref(), ref_array.as_ref(), abs_all <= 1e-3);
}
