pub type FP = f64;
pub use std::f64::EPSILON as FP_EPSILON;
pub use std::f64::MIN as FP_MIN;

pub type Mat = MatGen<Vec<FP>>;
pub type MatSlice<'a> = MatGen<&'a [FP]>;
pub type MatSliMu<'a> = MatGen<&'a mut[FP]>;

use std::cmp::PartialEq;
use std::ops::{Range, RangeBounds, Bound};
use std::ops::{Neg, Add, Mul, Sub, Div};
use std::ops::{Index, IndexMut};
use std::fmt;

pub trait View {
    fn get_ref(&self) -> &[FP];
    fn get_mut(&mut self) -> &mut[FP];
    fn get_len(&self) -> usize;
    fn get_own(self) -> Vec<FP>;
    fn is_own(&self) -> bool;
}

impl View for Vec<FP>
{
    fn get_ref(&self) -> &[FP]
    {
        self.as_ref()
    }
    fn get_mut(&mut self) -> &mut[FP]
    {
        self.as_mut()
    }
    fn get_len(&self) -> usize
    {
        self.len()
    }
    fn get_own(self) -> Vec<FP>
    {
        self
    }
    fn is_own(&self) -> bool
    {
        true
    }
}

impl View for &[FP]
{
    fn get_ref(&self) -> &[FP]
    {
        self
    }
    fn get_mut(&mut self) -> &mut[FP]
    {
        panic!("cannot borrow immutable as mutable");
    }
    fn get_len(&self) -> usize
    {
        self.len()
    }
    fn get_own(self) -> Vec<FP>
    {
        panic!("cannot own immutable");
    }
    fn is_own(&self) -> bool
    {
        false
    }
}

impl View for &mut[FP]
{
    fn get_ref(&self) -> &[FP]
    {
        self
    }
    fn get_mut(&mut self) -> &mut[FP]
    {
        self
    }
    fn get_len(&self) -> usize
    {
        self.len()
    }
    fn get_own(self) -> Vec<FP>
    {
        panic!("cannot own mutable");
    }
    fn is_own(&self) -> bool
    {
        false
    }
}

#[derive(Debug)]
pub struct MatGen<V: View>
{
    nrows: usize,
    ncols: usize,
    //
    offset: usize,
    stride: usize,
    //
    transposed: bool,
    //
    view: V
}

impl<V: View> MatGen<V>
{
    // private helper methods
    fn h_index(&self, index: (usize, usize)) -> usize
    {
        if !self.transposed {
            self.offset + self.stride * index.1 + index.0
        }
        else {
            self.offset + self.stride * index.0 + index.1
        }
    }
    //
    fn h_bound<RR, CR>(&self, rows: RR, cols: CR) -> (Range<usize>, Range<usize>)
    where RR: RangeBounds<usize>, CR: RangeBounds<usize>
    {
        let row_b = match rows.start_bound() {
            Bound::Unbounded => 0,
            Bound::Included(&i) => i,
            Bound::Excluded(&i) => i + 1
        };

        let row_e = match rows.end_bound() {
            Bound::Unbounded => if !self.transposed {self.nrows} else {self.ncols},
            Bound::Included(&i) => i + 1,
            Bound::Excluded(&i) => i
        };

        let col_b = match cols.start_bound() {
            Bound::Unbounded => 0,
            Bound::Included(&i) => i,
            Bound::Excluded(&i) => i + 1
        };

        let col_e = match cols.end_bound() {
            Bound::Unbounded => if !self.transposed {self.ncols} else {self.nrows},
            Bound::Included(&i) => i + 1,
            Bound::Excluded(&i) => i
        };

        if !self.transposed {
            (Range{start: row_b, end: row_e}, Range{start: col_b, end: col_e})
        }
        else {
            (Range{start: col_b, end: col_e}, Range{start: row_b, end: row_e})
        }
    }
    //
    fn h_own(self) -> Mat
    {
        if self.view.is_own() {
            MatGen {
                nrows: self.nrows,
                ncols: self.ncols,
                offset: self.offset,
                stride: self.stride,
                transposed: self.transposed,
                view: self.view.get_own()
            }
        }
        else {
            self.clone()
        }
    }
    //
    // new methods
    pub fn new(nrows: usize, ncols: usize) -> Mat
    {
        MatGen {
            nrows,
            ncols,
            offset: 0,
            stride: nrows,
            transposed: false,
            view: vec![0.0; nrows * ncols]
        }
    }
    //
    pub fn new_like<V2: View>(mat: &MatGen<V2>) -> Mat
    {
        let (nrows, ncols) = mat.size();

        Mat::new(nrows, ncols)
    }
    //
    pub fn new_vec(nrows: usize) -> Mat
    {
        Mat::new(nrows, 1)
    }
    //
    // slice methods
    pub fn slice<RR, CR>(&self, rows: RR, cols: CR) -> MatSlice
    where RR: RangeBounds<usize>,  CR: RangeBounds<usize>
    {
        let (row_range, col_range) = self.h_bound(rows, cols);

        MatGen {
            nrows: row_range.end - row_range.start,
            ncols: col_range.end - col_range.start,
            offset: self.offset + self.stride * col_range.start + row_range.start,
            stride: self.stride,
            transposed: self.transposed,
            view: self.view.get_ref()
        }
    }
    //
    pub fn slice_mut<'b, RR, CR>(&'b mut self, rows: RR, cols: CR) -> MatSliMu
    where RR: RangeBounds<usize>,  CR: RangeBounds<usize>
    {
        let (row_range, col_range) = self.h_bound(rows, cols);

        MatGen {
            nrows: row_range.end - row_range.start,
            ncols: col_range.end - col_range.start,
            offset: self.offset + self.stride * col_range.start + row_range.start,
            stride: self.stride,
            transposed: self.transposed,
            view: self.view.get_mut()
        }
    }
    //
    pub fn rows<RR>(&self, rows: RR) -> MatSlice
    where RR: RangeBounds<usize>
    {
        self.slice(rows, ..)
    }
    //
    pub fn cols<CR>(&self, cols: CR) -> MatSlice
    where CR: RangeBounds<usize>
    {
        self.slice(.., cols)
    }
    //
    pub fn row(&self, r: usize) -> MatSlice
    {
        self.rows(r ..= r)
    }
    //
    pub fn col(&self, c: usize) -> MatSlice
    {
        self.cols(c ..= c)
    }
    //
    pub fn rows_mut<RR>(&mut self, rows: RR) -> MatSliMu
    where RR: RangeBounds<usize>
    {
        self.slice_mut(rows, ..)
    }
    //
    pub fn cols_mut<CR>(&mut self, cols: CR) -> MatSliMu
    where CR: RangeBounds<usize>
    {
        self.slice_mut(.., cols)
    }
    //
    pub fn row_mut(&mut self, r: usize) -> MatSliMu
    {
        self.rows_mut(r ..= r)
    }
    //
    pub fn col_mut(&mut self, c: usize) -> MatSliMu
    {
        self.cols_mut(c ..= c)
    }
    //
    pub fn t(&self) -> MatSlice
    {
        MatGen {
            nrows: self.nrows,
            ncols: self.ncols,
            offset: self.offset,
            stride: self.stride,
            transposed: !self.transposed,
            view: self.view.get_ref()
        }
    }
    //
    pub fn t_mut(&mut self) -> MatSliMu
    {
        MatGen {
            nrows: self.nrows,
            ncols: self.ncols,
            offset: self.offset,
            stride: self.stride,
            transposed: !self.transposed,
            view: self.view.get_mut()
        }
    }
    //
    // set methods
    pub fn set_by<F>(mut self, mut f: F) -> MatGen<V>
    where F: FnMut(usize, usize) -> FP
    {
        let (nrows, ncols) = self.size();

        for c in 0 .. ncols {
            for r in 0 .. nrows {
                self[(r, c)] = f(r, c);
            }
        }
        self
    }
    //
    pub fn set_iter<'b, T>(mut self, iter: T) -> MatGen<V>
    where T: IntoIterator<Item=&'b FP>
    {
        let (nrows, ncols) = self.size();
        let mut i = iter.into_iter();

        // NOTE: contents of iter is row-wise
        for r in 0 .. nrows {
            for c in 0 .. ncols {
                self[(r, c)] = *i.next().unwrap_or(&0.);
            }
        }
        self
    }
    //
    pub fn set_eye(self) -> MatGen<V>
    {
        self.set_by(|r, c| if r == c {1.} else {0.})
    }
    //
    pub fn set_all(self, value: FP) -> MatGen<V>
    {
        self.set_by(|_, _| value)
    }
    //
    pub fn set_t(mut self) -> MatGen<V>
    {
        self.transposed = !self.transposed;
        self
    }
    //
    // clone methods
    pub fn clone(&self) -> Mat
    {
        // NOTE: this is not std::clone::Clone trait

        let (l_nrows, l_ncols) = self.size();
        let sz = self.view.get_len();

        if sz == l_nrows * l_ncols {
            MatGen {
                nrows: self.nrows,
                ncols: self.ncols,
                offset: self.offset,
                stride: self.stride,
                transposed: self.transposed,
                view: self.view.get_ref().to_vec()
            }
        }
        else {
            let mut mat = Mat::new(l_nrows, l_ncols);
            mat.assign(self);
            mat
        }
    }
    //
    pub fn clone_diag(&self) -> Mat
    {
        let (l_nrows, l_ncols) = self.size();
        assert_eq!(l_ncols, 1);

        let mut mat = Mat::new(l_nrows, l_nrows);

        for r in 0 .. l_nrows {
            mat[(r, r)] = self[(r, 0)];
        }

        mat
    }
    //
    // assign methods
    pub fn assign_by<F>(&mut self, f: F)
    where F: Fn(usize, usize) -> Option<FP>
    {
        let (l_nrows, l_ncols) = self.size();

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                if let Some(value) = f(r, c) {
                    self[(r, c)] = value;
                }
            }
        }
    }
    //
    pub fn assign_all(&mut self, value: FP)
    {
        self.assign_by(|_, _| Some(value));
    }
    //
    pub fn assign<V2: View>(&mut self, rhs: &MatGen<V2>)
    {
        let (l_nrows, l_ncols) = self.size();
        let (r_nrows, r_ncols) = rhs.size();

        assert_eq!(l_nrows, r_nrows);
        assert_eq!(l_ncols, r_ncols);
        
        self.assign_by(|r, c| Some(rhs[(r, c)]));
    }
    //
    // norm methods
    pub fn norm_p2sq(&self) -> FP
    {
        let (l_nrows, l_ncols) = self.size();

        let mut sum = 0.;

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                sum += self[(r, c)] * self[(r, c)];
            }
        }

        sum
    }
    //
    pub fn norm_p2(&self) -> FP
    {
        FP::sqrt(self.norm_p2sq())
    }
    //
    // search methods
    pub fn max(&self) -> (usize, usize, FP)
    {
        let (l_nrows, l_ncols) = self.size();
        assert_ne!(l_nrows, 0);
        assert_ne!(l_ncols, 0);

        let mut mr: usize = 0;
        let mut mc: usize = 0;
        let mut m = self[(0, 0)];

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                if self[(r, c)] > m {
                    m = self[(r, c)];
                    mr = r;
                    mc = c;
                }
            }
        }

        (mr, mc, m)
    }
    //
    pub fn min(&self) -> (usize, usize, FP)
    {
        let (l_nrows, l_ncols) = self.size();
        assert_ne!(l_nrows, 0);
        assert_ne!(l_ncols, 0);
        
        let mut mr: usize = 0;
        let mut mc: usize = 0;
        let mut m = self[(0, 0)];

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                if self[(r, c)] < m {
                    m = self[(r, c)];
                    mr = r;
                    mc = c;
                }
            }
        }

        (mr, mc, m)
    }
    //
    // size methods
    pub fn size(&self) -> (usize, usize)
    {
        if !self.transposed {
            (self.nrows, self.ncols)
        }
        else {
            (self.ncols, self.nrows)
        }
    }
}

//

impl<V: View> Index<(usize, usize)> for MatGen<V>
{
    type Output = FP;
    fn index(&self, index: (usize, usize)) -> &FP
    {
        let i = self.h_index(index);

        &self.view.get_ref()[i]
    }
}

impl<V: View> IndexMut<(usize, usize)> for MatGen<V>
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut FP
    {
        let i = self.h_index(index);

        &mut self.view.get_mut()[i]
    }
}

//

impl<V: View> fmt::Display for MatGen<V>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error>
    {
        let (l_nrows, l_ncols) = self.size();

        writeln!(f, "[")?;
        for r in 0 .. l_nrows {
            for c in 0 .. l_ncols {
                write!(f, "  {:.3e},", self[(r, c)])?;
            }
            writeln!(f)?;
        }
        write!(f, "]")
    }
}

//

impl<V: View, V2: View> PartialEq<MatGen<V2>> for MatGen<V>
{
    fn eq(&self, other: &MatGen<V2>) -> bool
    {
        let (l_nrows, l_ncols) = self.size();

        if (l_nrows, l_ncols) != other.size() {
            return false;
        }

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                if self[(r, c)] != other[(r, c)] {
                    return false;
                }
            }
        }

        true
    }
}

//

pub trait MatAcc
{
    fn acc_size(&self) -> (usize, usize);
    fn acc_get(&self, row: usize, col: usize) -> FP;
}

impl<V: View> MatAcc for MatGen<V>
{
    fn acc_size(&self) -> (usize, usize)
    {
        self.size()
    }
    //
    fn acc_get(&self, row: usize, col: usize) -> FP
    {
        self[(row, col)]
    }
}

impl<V: View> MatAcc for &MatGen<V>
{
    fn acc_size(&self) -> (usize, usize)
    {
        (*self).acc_size()
    }
    //
    fn acc_get(&self, row: usize, col: usize) -> FP
    {
        (*self).acc_get(row, col)
    }
}

//
impl<V: View> Neg for MatGen<V>
{
    type Output = Mat;

    fn neg(self) -> Mat
    {
        let mut mat = self.h_own();
        let (l_nrows, l_ncols) = mat.size();

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                mat[(r, c)] = -mat[(r, c)];
            }
        }

        mat
    }
}

impl<V: View> Neg for &MatGen<V>
{
    type Output = Mat;

    fn neg(self) -> Mat
    {
        self.clone().neg()
    }
}

//

impl<V: View, T: MatAcc> Add<T> for MatGen<V>
{
    type Output = Mat;

    fn add(self, rhs: T) -> Mat
    {
        let mut mat = self.h_own();
        let (l_nrows, l_ncols) = mat.size();

        assert_eq!((l_nrows, l_ncols), rhs.acc_size());

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                mat[(r, c)] += rhs.acc_get(r, c);
            }
        }

        mat
    }
}

impl<V: View, T: MatAcc> Add<T> for &MatGen<V>
{
    type Output = Mat;

    fn add(self, rhs: T) -> Mat
    {
        self.clone().add(rhs)
    }
}

impl<V: View> Add<FP> for MatGen<V>
{
    type Output = Mat;

    fn add(self, rhs: FP) -> Mat
    {
        let mut mat = self.h_own();
        let (l_nrows, l_ncols) = mat.size();

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                mat[(r, c)] += rhs;
            }
        }

        mat
    }
}

impl<V: View> Add<FP> for &MatGen<V>
{
    type Output = Mat;

    fn add(self, rhs: FP) -> Mat
    {
        self.clone().add(rhs)
    }
}

impl<V: View> Add<MatGen<V>> for FP
{
    type Output = Mat;

    fn add(self, rhs: MatGen<V>) -> Mat
    {
        rhs.add(self)
    }
}

impl<V: View> Add<&MatGen<V>> for FP
{
    type Output = Mat;

    fn add(self, rhs: &MatGen<V>) -> Mat
    {
        rhs.add(self)
    }
}

//

impl<V: View, T: MatAcc> Sub<T> for MatGen<V>
{
    type Output = Mat;

    fn sub(self, rhs: T) -> Mat
    {
        let mut mat = self.h_own();
        let (l_nrows, l_ncols) = mat.size();

        assert_eq!((l_nrows, l_ncols), rhs.acc_size());

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                mat[(r, c)] -= rhs.acc_get(r, c);
            }
        }

        mat
    }
}

impl<V: View, T: MatAcc> Sub<T> for &MatGen<V>
{
    type Output = Mat;

    fn sub(self, rhs: T) -> Mat
    {
        self.clone().sub(rhs)
    }
}

impl<V: View> Sub<FP> for MatGen<V>
{
    type Output = Mat;

    fn sub(self, rhs: FP) -> Mat
    {
        let mut mat = self.h_own();
        let (l_nrows, l_ncols) = mat.size();

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                mat[(r, c)] -= rhs;
            }
        }

        mat
    }
}

impl<V: View> Sub<FP> for &MatGen<V>
{
    type Output = Mat;

    fn sub(self, rhs: FP) -> Mat
    {
        self.clone().sub(rhs)
    }
}

impl<V: View> Sub<MatGen<V>> for FP
{
    type Output = Mat;

    fn sub(self, rhs: MatGen<V>) -> Mat
    {
        rhs.neg().add(self)
    }
}

impl<V: View> Sub<&MatGen<V>> for FP
{
    type Output = Mat;

    fn sub(self, rhs: &MatGen<V>) -> Mat
    {
        rhs.neg().add(self)
    }
}

//

impl<V: View, T: MatAcc> Mul<T> for MatGen<V>
{
    type Output = Mat;

    fn mul(self, rhs: T) -> Mat
    {
        (&self).mul(rhs)
    }
}

impl<V: View, T: MatAcc> Mul<T> for &MatGen<V>
{
    type Output = Mat;

    fn mul(self, rhs: T) -> Mat
    {
        let (l_nrows, l_ncols) = self.size();
        let (r_nrows, r_ncols) = rhs.acc_size();

        assert_eq!(l_ncols, r_nrows);

        let mut mat = Mat::new(l_nrows, r_ncols);

        for c in 0 .. r_ncols {
            for r in 0 .. l_nrows {
                let mut v: FP = 0.0;
                for k in 0 .. l_ncols {
                    v += self[(r, k)] * rhs.acc_get(k, c);
                }
                mat[(r, c)] = v;
            }
        }

        mat
    }
}

impl<V: View> Mul<FP> for MatGen<V>
{
    type Output = Mat;

    fn mul(self, rhs: FP) -> Mat
    {
        let mut mat = self.h_own();
        let (l_nrows, l_ncols) = mat.size();

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                mat[(r, c)] *= rhs;
            }
        }

        mat
    }
}

impl<V: View> Mul<FP> for &MatGen<V>
{
    type Output = Mat;

    fn mul(self, rhs: FP) -> Mat
    {
        self.clone().mul(rhs)
    }
}

impl<V: View> Mul<MatGen<V>> for FP
{
    type Output = Mat;

    fn mul(self, rhs: MatGen<V>) -> Mat
    {
        rhs.mul(self)
    }
}

impl<V: View> Mul<&MatGen<V>> for FP
{
    type Output = Mat;

    fn mul(self, rhs: &MatGen<V>) -> Mat
    {
        rhs.mul(self)
    }
}

//

impl<V: View> Div<FP> for MatGen<V>
{
    type Output = Mat;

    fn div(self, rhs: FP) -> Mat
    {
        let mut mat = self.h_own();
        let (l_nrows, l_ncols) = mat.size();

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                mat[(r, c)] /= rhs;
            }
        }

        mat
    }
}

impl<V: View> Div<FP> for &MatGen<V>
{
    type Output = Mat;

    fn div(self, rhs: FP) -> Mat
    {
        self.clone().div(rhs)
    }
}

//

pub const XOR64_INIT: u64 = 88172645463325252;

pub fn xor64(state: &mut u64) -> FP
{
    const MAX: FP = (1_u128 << 64) as FP;
    *state = *state ^ (*state << 7);
    *state = *state ^ (*state >> 9);

    // [0.0, 1.0)
    (*state as FP) / MAX
}

//

#[test]
fn test_set()
{
    {
        let a = Mat::new(3, 3).set_eye();
        let b = Mat::new_like(&a).set_iter(&[
            1., 0., 0.,
            0., 1., 0.,
            0., 0., 1.
        ]);
        assert_eq!(a, b);
    }
    {
        let a = Mat::new(2, 4).set_by(|r, c| (r * 4 + c) as FP);
        let b = Mat::new_like(&a).set_iter(&[
            0., 1., 2., 3.,
            4., 5., 6., 7.
        ]);
        assert_eq!(a, b);
    }
}

#[test]
fn test_misc()
{
    {
        let a = Mat::new_vec(3);
        let a = a.t();
        let b = Mat::new(1, 3);
        assert_eq!(a, b);
    }
    {
        let a = Mat::new_vec(3).set_t();
        let b = Mat::new(1, 3);
        assert_eq!(a, b);
    }
    {
        let mut r = XOR64_INIT;
        let mut a = Mat::new(4, 4);
        let b = Mat::new_like(&a).set_by(|_, _| xor64(&mut r));
        a.assign(&b);
        assert_eq!(a, b);
    }
}

#[test]
fn test_slice()
{
    {
        let a = Mat::new(4, 4).set_eye();
        let a = a.slice(1 ..= 2, 1 ..= 2);
        let b = Mat::new(2, 2).set_eye();
        assert_eq!(a, b);
    }
    {
        let mut a = Mat::new(4, 4).set_eye();
        let b = Mat::new(4, 4).set_iter(&[
            1., 0., 0., 0.,
            0., 2., 2., 0.,
            0., 2., 2., 0.,
            0., 0., 0., 1.
        ]);
        let mut a1 = a.slice_mut(1 ..= 2, 1 ..= 2);
        let a2 = Mat::new(2, 2).set_all(2.);
        a1.assign(&a2);
        assert_eq!(a, b);
    }
    {
        let mut a = Mat::new(4, 4).set_eye();
        let b = Mat::new(4, 4).set_iter(&[
            0., 0., 0., 0.,
            0., 1., 0., 0.,
            0., 0., 1., 0.,
            1., 0., 0., 1.
        ]);
        let a1 = a.col(3).clone();
        a.col_mut(0).assign(&a1);
        assert_eq!(a, b);
    }
}

#[test]
fn test_ops()
{
    {
        let a = Mat::new(2, 2).set_eye();
        let b = Mat::new(2, 2).set_iter(&[
            -1., 0.,
            0., -1.
        ]);
        let c = -a;
        assert_eq!(c, b);
    }
    {
        let a = Mat::new(2, 2).set_eye();
        let b = Mat::new(2, 2).set_iter(&[
            -1., 0.,
            0., -1.
        ]);
        let c = -&a;
        assert_eq!(c, b);
        println!("{}", a);
    }
    {
        let a1 = Mat::new(2, 2).set_eye();
        let a2 = Mat::new(2, 2).set_all(1.);
        let b = Mat::new(2, 2).set_iter(&[
            2., 1.,
            1., 2.
        ]);
        let c = a1 + a2;
        assert_eq!(c, b);
    }
    {
        let a1 = Mat::new(2, 2).set_eye();
        let a2 = Mat::new(2, 2).set_all(1.);
        let b = Mat::new(2, 2).set_iter(&[
            2., 1.,
            1., 2.
        ]);
        let c = &a1 + &a2;
        assert_eq!(c, b);
        println!("{}", a1);
    }
}
