use crate::matop::{MatType, MatOp};
use crate::linalgex::LinAlgEx;
use core::ops::{Index, IndexMut};
use core::marker::PhantomData;

//

#[derive(Debug, Clone)]
pub struct MatBuild<L>
where L: LinAlgEx<f64>
{
    _ph_l: PhantomData<L>,
    typ: MatType,
    array: Vec<f64>,
}

impl<L> MatBuild<L>
where L: LinAlgEx<f64>
{
    pub fn new(typ: MatType) -> Self
    {
        MatBuild {
            _ph_l: PhantomData,
            typ,
            array: vec![0.; typ.len()],
        }
    }

    pub fn typ(&self) -> &MatType
    {
        &self.typ
    }

    pub fn set_by_fn<F>(&mut self, mut func: F)
    where F: FnMut(usize, usize) -> f64
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
    pub fn by_fn<F>(mut self, func: F) -> Self
    where F: FnMut(usize, usize) -> f64
    {
        self.set_by_fn(func);
        self
    }

    pub fn set_iter_colmaj<'a, T>(&mut self, iter: T)
    where T: IntoIterator<Item=&'a f64>
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
    pub fn iter_colmaj<'a, T>(mut self, iter: T) -> Self
    where T: IntoIterator<Item=&'a f64>
    {
        self.set_iter_colmaj(iter);
        self
    }

    pub fn set_iter_rowmaj<'a, T>(&mut self, iter: T)
    where T: IntoIterator<Item=&'a f64>
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
    pub fn iter_rowmaj<'a, T>(mut self, iter: T) -> Self
    where T: IntoIterator<Item=&'a f64>
    {
        self.set_iter_rowmaj(iter);
        self
    }

    pub fn set_scale(&mut self, alpha: f64)
    {
        L::scale(alpha, self.as_mut());
    }
    pub fn scale(mut self, alpha: f64) -> Self
    {
        self.set_scale(alpha);
        self
    }

    pub fn set_scale_nondiag(&mut self, alpha: f64)
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
    pub fn scale_nondiag(mut self, alpha: f64) -> Self
    {
        self.set_scale_nondiag(alpha);
        self
    }

    pub fn set_reshape_vec(&mut self)
    {
        let sz = self.as_ref().len();
        self.typ = MatType::General(sz, 1);
    }
    pub fn reshape_colvec(mut self) -> Self
    {
        self.set_reshape_vec();
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

impl<L> Index<(usize, usize)> for MatBuild<L>
where L: LinAlgEx<f64>
{
    type Output = f64;
    fn index(&self, index: (usize, usize)) -> &f64
    {
        let i = self.index(index);

        &self.array[i]
    }
}

impl<L> IndexMut<(usize, usize)> for MatBuild<L>
where L: LinAlgEx<f64>
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut f64
    {
        let i = self.index(index);

        &mut self.array[i]
    }
}

impl<L> AsRef<[f64]> for MatBuild<L>
where L: LinAlgEx<f64>
{
    fn as_ref(&self) -> &[f64]
    {
        &self.array
    }
}

impl<L> AsMut<[f64]> for MatBuild<L>
where L: LinAlgEx<f64>
{
    fn as_mut(&mut self) -> &mut[f64]
    {
        &mut self.array
    }
}

impl<'a, L> From<&'a MatBuild<L>> for MatOp<'a, L, f64>
where L: LinAlgEx<f64>
{
    fn from(m: &'a MatBuild<L>) -> Self
    {
        MatOp::new(m.typ, m.as_ref())
    }
}


#[test]
fn test_matbuild1() {
    use float_eq::assert_float_eq;
    use crate::f64_lapack::F64LAPACK;

    type AMatBuild = MatBuild<F64LAPACK>;

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
