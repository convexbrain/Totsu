pub type FP = f64;
pub use std::f64::EPSILON as FP_EPSILON;
pub use std::f64::MIN as FP_MIN;

pub type MatOwn = Mat<Vec<FP>>;

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
pub struct Mat<V: View>
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

impl<V: View> Mat<V>
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
    fn h_own(self) -> MatOwn
    {
        if self.view.is_own() {
            Mat {
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
    pub fn new(nrows: usize, ncols: usize) -> MatOwn
    {
        Mat {
            nrows,
            ncols,
            offset: 0,
            stride: nrows,
            transposed: false,
            view: vec![0.0; nrows * ncols]
        }
    }
    //
    pub fn new_like<V2: View>(mat: &Mat<V2>) -> MatOwn
    {
        let (nrows, ncols) = mat.size();

        MatOwn::new(nrows, ncols)
    }
    //
    pub fn new_vec(nrows: usize) -> MatOwn
    {
        MatOwn::new(nrows, 1)
    }
    //
    // refer methods
    pub fn slice<RR, CR>(&self, rows: RR, cols: CR) -> Mat<&[FP]>
    where RR: RangeBounds<usize>,  CR: RangeBounds<usize>
    {
        let (row_range, col_range) = self.h_bound(rows, cols);

        Mat {
            nrows: row_range.end - row_range.start,
            ncols: col_range.end - col_range.start,
            offset: self.offset + self.stride * col_range.start + row_range.start,
            stride: self.stride,
            transposed: self.transposed,
            view: self.view.get_ref()
        }
    }
    //
    pub fn slice_mut<'b, RR, CR>(&'b mut self, rows: RR, cols: CR) -> Mat<&mut[FP]>
    where RR: RangeBounds<usize>,  CR: RangeBounds<usize>
    {
        let (row_range, col_range) = self.h_bound(rows, cols);

        Mat {
            nrows: row_range.end - row_range.start,
            ncols: col_range.end - col_range.start,
            offset: self.offset + self.stride * col_range.start + row_range.start,
            stride: self.stride,
            transposed: self.transposed,
            view: self.view.get_mut()
        }
    }
    //
    pub fn row(&self, r: usize) -> Mat<&[FP]>
    {
        self.slice(r ..= r, ..)
    }
    //
    pub fn col(&self, c: usize) -> Mat<&[FP]>
    {
        self.slice(.., c ..= c)
    }
    //
    pub fn row_mut(&mut self, r: usize) -> Mat<&mut[FP]>
    {
        self.slice_mut(r ..= r, ..)
    }
    //
    pub fn col_mut(&mut self, c: usize) -> Mat<&mut[FP]>
    {
        self.slice_mut(.., c ..= c)
    }
    //
    pub fn t(&self) -> Mat<&[FP]>
    {
        Mat {
            nrows: self.nrows,
            ncols: self.ncols,
            offset: self.offset,
            stride: self.stride,
            transposed: !self.transposed,
            view: self.view.get_ref()
        }
    }
    //
    pub fn t_mut(&mut self) -> Mat<&mut[FP]>
    {
        Mat {
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
    pub fn set_by<F>(mut self, mut f: F) -> Mat<V>
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
    pub fn set_iter<'b, T>(mut self, iter: T) -> Mat<V>
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
    pub fn set_eye(self) -> Mat<V>
    {
        self.set_by(|r, c| {if r == c {1.} else {0.}})
    }
    //
    pub fn set_all(self, value: FP) -> Mat<V>
    {
        self.set_by(|_, _| {value})
    }
    //
    pub fn set_t(mut self) -> Mat<V>
    {
        self.transposed = !self.transposed;
        self
    }
    //
    // clone methods
    pub fn clone(&self) -> MatOwn
    {
        // NOTE: this is not std::clone::Clone trait

        let sz = self.view.get_len();

        if sz == self.nrows * self.ncols {
            Mat {
                nrows: self.nrows,
                ncols: self.ncols,
                offset: self.offset,
                stride: self.stride,
                transposed: self.transposed,
                view: self.view.get_ref().to_vec()
            }
        }
        else {
            let (l_nrows, l_ncols) = self.size();
            let mut mat = MatOwn::new(l_nrows, l_ncols);
            mat.assign(self);
            mat
        }
    }
    //
    pub fn clone_diag(&self) -> MatOwn
    {
        let (l_nrows, l_ncols) = self.size();
        assert_eq!(l_ncols, 1);

        let mut mat = MatOwn::new(l_nrows, l_nrows);

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
    pub fn assign<V2: View>(&mut self, rhs: &Mat<V2>)
    {
        let (l_nrows, l_ncols) = self.size();
        let (r_nrows, r_ncols) = rhs.size();

        assert_eq!(l_nrows, r_nrows);
        assert_eq!(l_ncols, r_ncols);
        
        self.assign_by(|r, c| {Some(rhs[(r, c)])});
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
}

//

impl<V: View> Index<(usize, usize)> for Mat<V>
{
    type Output = FP;
    fn index(&self, index: (usize, usize)) -> &FP
    {
        let i = self.h_index(index);

        &self.view.get_ref()[i]
    }
}

impl<V: View> IndexMut<(usize, usize)> for Mat<V>
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut FP
    {
        let i = self.h_index(index);

        &mut self.view.get_mut()[i]
    }
}

//

impl<V: View> fmt::Display for Mat<V>
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

impl<V: View, V2: View> PartialEq<Mat<V2>> for Mat<V>
{
    fn eq(&self, other: &Mat<V2>) -> bool
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

impl<V: View> MatAcc for Mat<V>
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

impl<V: View> MatAcc for &Mat<V>
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
impl<V: View> Neg for Mat<V>
{
    type Output = MatOwn;

    fn neg(self) -> MatOwn
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

impl<V: View> Neg for &Mat<V>
{
    type Output = MatOwn;

    fn neg(self) -> MatOwn
    {
        self.clone().neg()
    }
}

//

impl<V: View, T: MatAcc> Add<T> for Mat<V>
{
    type Output = MatOwn;

    fn add(self, rhs: T) -> MatOwn
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

impl<V: View, T: MatAcc> Add<T> for &Mat<V>
{
    type Output = MatOwn;

    fn add(self, rhs: T) -> MatOwn
    {
        self.clone().add(rhs)
    }
}

impl<V: View> Add<FP> for Mat<V>
{
    type Output = MatOwn;

    fn add(self, rhs: FP) -> MatOwn
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

impl<V: View> Add<FP> for &Mat<V>
{
    type Output = MatOwn;

    fn add(self, rhs: FP) -> MatOwn
    {
        self.clone().add(rhs)
    }
}

impl<V: View> Add<Mat<V>> for FP
{
    type Output = MatOwn;

    fn add(self, rhs: Mat<V>) -> MatOwn
    {
        rhs.add(self)
    }
}

impl<V: View> Add<&Mat<V>> for FP
{
    type Output = MatOwn;

    fn add(self, rhs: &Mat<V>) -> MatOwn
    {
        rhs.add(self)
    }
}

//

impl<V: View, T: MatAcc> Sub<T> for Mat<V>
{
    type Output = MatOwn;

    fn sub(self, rhs: T) -> MatOwn
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

impl<V: View, T: MatAcc> Sub<T> for &Mat<V>
{
    type Output = MatOwn;

    fn sub(self, rhs: T) -> MatOwn
    {
        self.clone().sub(rhs)
    }
}

impl<V: View> Sub<FP> for Mat<V>
{
    type Output = MatOwn;

    fn sub(self, rhs: FP) -> MatOwn
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

impl<V: View> Sub<FP> for &Mat<V>
{
    type Output = MatOwn;

    fn sub(self, rhs: FP) -> MatOwn
    {
        self.clone().sub(rhs)
    }
}

impl<V: View> Sub<Mat<V>> for FP
{
    type Output = MatOwn;

    fn sub(self, rhs: Mat<V>) -> MatOwn
    {
        rhs.neg().add(self)
    }
}

impl<V: View> Sub<&Mat<V>> for FP
{
    type Output = MatOwn;

    fn sub(self, rhs: &Mat<V>) -> MatOwn
    {
        rhs.neg().add(self)
    }
}

//

impl<V: View, T: MatAcc> Mul<T> for Mat<V>
{
    type Output = MatOwn;

    fn mul(self, rhs: T) -> MatOwn
    {
        (&self).mul(rhs)
    }
}

impl<V: View, T: MatAcc> Mul<T> for &Mat<V>
{
    type Output = MatOwn;

    fn mul(self, rhs: T) -> MatOwn
    {
        let (l_nrows, l_ncols) = self.size();
        let (r_nrows, r_ncols) = rhs.acc_size();

        assert_eq!(l_ncols, r_nrows);

        let mut mat = MatOwn::new(l_nrows, r_ncols);

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

impl<V: View> Mul<FP> for Mat<V>
{
    type Output = MatOwn;

    fn mul(self, rhs: FP) -> MatOwn
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

impl<V: View> Mul<FP> for &Mat<V>
{
    type Output = MatOwn;

    fn mul(self, rhs: FP) -> MatOwn
    {
        self.clone().mul(rhs)
    }
}

impl<V: View> Mul<Mat<V>> for FP
{
    type Output = MatOwn;

    fn mul(self, rhs: Mat<V>) -> MatOwn
    {
        rhs.mul(self)
    }
}

impl<V: View> Mul<&Mat<V>> for FP
{
    type Output = MatOwn;

    fn mul(self, rhs: &Mat<V>) -> MatOwn
    {
        rhs.mul(self)
    }
}

//

impl<V: View> Div<FP> for Mat<V>
{
    type Output = MatOwn;

    fn div(self, rhs: FP) -> MatOwn
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

impl<V: View> Div<FP> for &Mat<V>
{
    type Output = MatOwn;

    fn div(self, rhs: FP) -> MatOwn
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
        let a = MatOwn::new(3, 3).set_eye();
        let b = MatOwn::new_like(&a).set_iter(&[
            1., 0., 0.,
            0., 1., 0.,
            0., 0., 1.
        ]);
        assert_eq!(a, b);
    }
    {
        let a = MatOwn::new(2, 4).set_by(|r, c| {(r * 4 + c) as FP});
        let b = MatOwn::new_like(&a).set_iter(&[
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
        let a = MatOwn::new_vec(3);
        let a = a.t();
        let b = MatOwn::new(1, 3);
        assert_eq!(a, b);
    }
    {
        let a = MatOwn::new_vec(3).set_t();
        let b = MatOwn::new(1, 3);
        assert_eq!(a, b);
    }
    {
        let mut r = XOR64_INIT;
        let mut a = MatOwn::new(4, 4);
        let b = MatOwn::new_like(&a).set_by(|_, _| {xor64(&mut r)});
        a.assign(&b);
        assert_eq!(a, b);
    }
}

#[test]
fn test_slice()
{
    {
        let a = MatOwn::new(4, 4).set_eye();
        let a = a.slice(1 ..= 2, 1 ..= 2);
        let b = MatOwn::new(2, 2).set_eye();
        assert_eq!(a, b);
    }
    {
        let mut a = MatOwn::new(4, 4).set_eye();
        let b = MatOwn::new(4, 4).set_iter(&[
            1., 0., 0., 0.,
            0., 2., 2., 0.,
            0., 2., 2., 0.,
            0., 0., 0., 1.
        ]);
        let mut a1 = a.slice_mut(1 ..= 2, 1 ..= 2);
        let a2 = MatOwn::new(2, 2).set_all(2.);
        a1.assign(&a2);
        assert_eq!(a, b);
    }
    {
        let mut a = MatOwn::new(4, 4).set_eye();
        let b = MatOwn::new(4, 4).set_iter(&[
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
        let a = MatOwn::new(2, 2).set_eye();
        let b = MatOwn::new(2, 2).set_iter(&[
            -1., 0.,
            0., -1.
        ]);
        let c = -a;
        assert_eq!(c, b);
    }
    {
        let a = MatOwn::new(2, 2).set_eye();
        let b = MatOwn::new(2, 2).set_iter(&[
            -1., 0.,
            0., -1.
        ]);
        let c = -&a;
        assert_eq!(c, b);
        println!("{}", a);
    }
    {
        let a1 = MatOwn::new(2, 2).set_eye();
        let a2 = MatOwn::new(2, 2).set_all(1.);
        let b = MatOwn::new(2, 2).set_iter(&[
            2., 1.,
            1., 2.
        ]);
        let c = a1 + a2;
        assert_eq!(c, b);
    }
    {
        let a1 = MatOwn::new(2, 2).set_eye();
        let a2 = MatOwn::new(2, 2).set_all(1.);
        let b = MatOwn::new(2, 2).set_iter(&[
            2., 1.,
            1., 2.
        ]);
        let c = &a1 + &a2;
        assert_eq!(c, b);
        println!("{}", a1);
    }
}
