use std::ops::{Index, IndexMut, Deref};
use num_traits::{Float, Zero};
use totsu_core::solver::SliceLike;
use totsu_core::{LinAlgEx, MatType, MatOp};

//

/// Matrix builder
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
/// 
/// Matrix struct which owns a `Vec` of data array and is able to be converted as [`totsu_core::MatOp`].
/// This struct relies on dynamic heap allocation.
#[derive(Debug, Clone)]
pub struct MatBuild<L: LinAlgEx>
{
    typ: MatType,
    array: Vec<L::F>,
}

impl<L: LinAlgEx> MatBuild<L>
{
    /// Creates an instance.
    /// 
    /// Returns the [`MatBuild`] instance with zero data.
    /// * `typ` is Matrix type and size.
    pub fn new(typ: MatType) -> Self
    {
        MatBuild {
            typ,
            array: vec![L::F::zero(); typ.len()],
        }
    }

    /// Size of the matrix.
    /// 
    /// Returns a tuple of a number of rows and columns.
    pub fn size(&self) -> (usize, usize)
    {
        self.typ.size()
    }

    /// Converted as [`totsu_core::MatOp`].
    /// 
    /// Returns the [`totsu_core::MatOp`] borrowing the internal data array.
    pub fn as_op(&self) -> MatOp<'_, L>
    {
        MatOp::new(self.typ, &self.array)
    }

    /// Checks if symmetric packed.
    /// 
    /// Returns `true` if [`MatType::SymPack`], `false` otherwise.
    pub fn is_sympack(&self) -> bool
    {
        if let MatType::SymPack(_) = self.typ {
            true
        }
        else {
            false
        }
    }

    /// Data by a function.
    /// 
    /// * `func` takes a row and a column of the matrix and returns data of each element.
    pub fn set_by_fn<M>(&mut self, mut func: M)
    where M: FnMut(usize, usize) -> L::F
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
    /// Builder pattern of [`MatBuild::set_by_fn`].
    pub fn by_fn<M>(mut self, func: M) -> Self
    where M: FnMut(usize, usize) -> L::F
    {
        self.set_by_fn(func);
        self
    }

    /// Data by an iterator in column-major.
    /// 
    /// * `iter` iterates matrix data in column-major.
    pub fn set_iter_colmaj<T, I>(&mut self, iter: T)
    where T: IntoIterator<Item=I>, I: Deref<Target=L::F>
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
    /// Builder pattern of [`MatBuild::set_iter_colmaj`].
    pub fn iter_colmaj<T, I>(mut self, iter: T) -> Self
    where T: IntoIterator<Item=I>, I: Deref<Target=L::F>
    {
        self.set_iter_colmaj(iter);
        self
    }

    /// Data by an iterator in row-major.
    /// 
    /// * `iter` iterates matrix data in row-major.
    pub fn set_iter_rowmaj<T, I>(&mut self, iter: T)
    where T: IntoIterator<Item=I>, I: Deref<Target=L::F>
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
    /// Builder pattern of [`MatBuild::set_iter_rowmaj`].
    pub fn iter_rowmaj<T, I>(mut self, iter: T) -> Self
    where T: IntoIterator<Item=I>, I: Deref<Target=L::F>
    {
        self.set_iter_rowmaj(iter);
        self
    }

    /// Scales by \\(\alpha\\).
    /// 
    /// * `alpha` is a scalar \\(\alpha\\).
    pub fn set_scale(&mut self, alpha: L::F)
    {
        L::scale(alpha, &mut L::Sl::new_mut(&mut self.array));

    }
    /// Builder pattern of [`MatBuild::set_scale`].
    pub fn scale(mut self, alpha: L::F) -> Self
    {
        self.set_scale(alpha);
        self
    }

    /// Scales by \\(\alpha\\) except diagonal elements.
    /// 
    /// * `alpha` is a scalar \\(\alpha\\).
    pub fn set_scale_nondiag(&mut self, alpha: L::F)
    {
        match self.typ {
            MatType::General(nr, nc) => {
                let n = nr.min(nc);
                for c in 0.. n - 1 {
                    let i = self.index((c, c));
                    let (_, spl) = self.array.split_at_mut(i + 1);
                    let (spl, _) = spl.split_at_mut(nc);
                    L::scale(alpha, &mut L::Sl::new_mut(spl));
                }
                let i = self.index((n, n));
                let (_, spl) = self.array.split_at_mut(i + 1);
                L::scale(alpha, &mut L::Sl::new_mut(spl));
            },
            MatType::SymPack(n) => {
                for c in 0.. n - 1 {
                    let i = self.index((c, c));
                    let ii = self.index((c + 1, c + 1));
                    let (_, spl) = self.array.split_at_mut(i + 1);
                    let (spl, _) = spl.split_at_mut(ii - i - 1);
                    L::scale(alpha, &mut L::Sl::new_mut(spl));
                }
            },
        }
    }
    /// Builder pattern of [`MatBuild::set_scale_nondiag`].
    pub fn scale_nondiag(mut self, alpha: L::F) -> Self
    {
        self.set_scale_nondiag(alpha);
        self
    }

    /// Reshapes the internal data array as it is into a one-column matrix.
    pub fn set_reshape_colvec(&mut self)
    {
        let sz = self.array.len();
        self.typ = MatType::General(sz, 1);
    }
    /// Builder pattern of [`MatBuild::set_reshape_colvec`].
    pub fn reshape_colvec(mut self) -> Self
    {
        self.set_reshape_colvec();
        self
    }

    /// Calculates and converts the matrix \\(A\\) to \\(A^{\frac12}\\).
    /// 
    /// The matrix shall belong to [`MatType::SymPack`].
    /// * `eps_zero` should be the same value as [`totsu_core::solver::SolverParam::eps_zero`].
    pub fn set_sqrt(&mut self, eps_zero: L::F)
    {
        match self.typ {
            MatType::General(_, _) => {
                unimplemented!()
            },
            MatType::SymPack(n) => {
                let mut work_vec = vec![L::F::zero(); L::map_eig_worklen(n)];
                let mut work = L::Sl::new_mut(&mut work_vec);

                let f0 = L::F::zero();
                L::map_eig(&mut L::Sl::new_mut(&mut self.array), None, eps_zero, &mut work, |e| {
                    if e > f0 {
                        Some(e.sqrt())
                    }
                    else {
                        None
                    }
                });
            }
        }
    }
    /// Builder pattern of [`MatBuild::set_sqrt`].
    pub fn sqrt(mut self, eps_zero: L::F) -> Self
    {
        self.set_sqrt(eps_zero);
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

//

impl<L: LinAlgEx> Index<(usize, usize)> for MatBuild<L>
{
    type Output = L::F;
    fn index(&self, index: (usize, usize)) -> &Self::Output
    {
        let i = self.index(index);

        &self.array[i]
    }
}

impl<L: LinAlgEx> IndexMut<(usize, usize)> for MatBuild<L>
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output
    {
        let i = self.index(index);

        &mut self.array[i]
    }
}

//

// used by examples
impl<L: LinAlgEx> AsRef<[L::F]> for MatBuild<L>
{
    fn as_ref(&self) -> &[L::F]
    {
        &self.array
    }
}

// used by examples
impl<L: LinAlgEx> AsMut<[L::F]> for MatBuild<L>
{
    fn as_mut(&mut self) -> &mut[L::F]
    {
        &mut self.array
    }
}

//

impl<L: LinAlgEx> core::fmt::Display for MatBuild<L>
where L::F: Float + core::fmt::LowerExp
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> Result<(), core::fmt::Error>
    {
        let (nr, nc) = self.size();
        if nr == 0 || nc == 0 {
            write!(f, "[ ]")?;
        }
        else {
            write!(f, "[ {:.3e}", self[(0, 0)])?;
            if nc > 2 {
                write!(f, " ...")?;
            }
            if nc > 1 {
                write!(f, " {:.3e}", self[(0, nc - 1)])?;
            }

            if nr > 2 {
                writeln!(f)?;
                write!(f, "  ...")?;
            }

            if nr > 1 {
                writeln!(f)?;
                write!(f, "  {:.3e}", self[(nr - 1, 0)])?;
                if nc > 2 {
                    write!(f, " ...")?;
                }
                if nc > 1 {
                    write!(f, " {:.3e}", self[(nr - 1, nc - 1)])?;
                }
            }
            write!(f, " ]")?;
        }

        write!(f, " ({} x {}) ", nr, nc)?;
        match self.typ {
            MatType::General(_, _) => write!(f, "General")?,
            MatType::SymPack(_) => write!(f, "Symmetric Packed")?,
        }

        Ok(())
    }
}

//

#[test]
fn test_matbuild1()
{
    use float_eq::assert_float_eq;
    use totsu_core::FloatGeneric;

    type L= FloatGeneric<f64>;

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

    let m = MatBuild::<L>::new(MatType::SymPack(5))
            .iter_colmaj(array)
            .scale_nondiag(1.4);

    let m_array: &[f64] = m.array.as_ref();
    assert_float_eq!(m_array, ref_array.as_ref(), abs_all <= 1e-3);
}
