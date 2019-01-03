pub type FP = f64;
pub use std::f64::EPSILON as FP_EPSILON;
pub use std::f64::MIN as FP_MIN;

use std::cmp::PartialEq;
use std::ops::{Range, RangeBounds, Bound};
use std::ops::{Neg, Add, Mul, Sub, Div};
use std::ops::{Index, IndexMut};
use std::fmt;

#[derive(Debug)]
enum View<'a>
{
    Own(Vec<FP>),
    Borrow(&'a [FP]),
    BorrowMut(&'a mut[FP])
}

impl<'a> Clone for View<'a>
{
    fn clone(&self) -> Self
    {
        match &self {
            View::Own(v) => View::Own(v.clone()),
            View::Borrow(v) => View::Own(v.to_vec()),
            View::BorrowMut(v) => View::Own(v.to_vec())
        }
    }
}

#[derive(Debug, Clone)]
pub struct Mat<'a>
{
    nrows: usize,
    ncols: usize,
    //
    offset: usize,
    stride: usize,
    //
    transposed: bool,
    diagonal: bool,
    //
    view: View<'a>
}

impl<'a> Mat<'a>
{
    pub fn new(nrows: usize, ncols: usize) -> Mat<'a>
    {
        Mat {
            nrows,
            ncols,
            offset: 0,
            stride: nrows,
            transposed: false,
            diagonal: false,
            view: View::Own(vec![0.0; nrows * ncols])
        }
    }
    //
    pub fn new1(nrows: usize) -> Mat<'a>
    {
        Mat::new(nrows, 1)
    }
    //
    fn tr_bound<RR, CR>(&self, rows: RR, cols: CR) -> (Range<usize>, Range<usize>)
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
    pub fn slice<RR, CR>(&self, rows: RR, cols: CR) -> Mat
    where RR: RangeBounds<usize>,  CR: RangeBounds<usize>
    {
        let (row_range, col_range) = self.tr_bound(rows, cols);

        let view = match &self.view {
            View::Own(v) => View::Borrow(&v),
            View::Borrow(v) => View::Borrow(&v),
            View::BorrowMut(v) => View::Borrow(&v)
        };

        Mat {
            nrows: row_range.end - row_range.start,
            ncols: col_range.end - col_range.start,
            offset: self.offset + self.stride * col_range.start + row_range.start,
            view,
            .. *self
        }
    }
    //
    pub fn slice_mut<RR, CR>(&mut self, rows: RR, cols: CR) -> Mat
    where RR: RangeBounds<usize>,  CR: RangeBounds<usize>
    {
        let (row_range, col_range) = self.tr_bound(rows, cols);

        let view = match &mut self.view {
            View::Own(v) => View::BorrowMut(v),
            View::Borrow(_) => panic!("cannot convert Borrow to BorrowMut"),
            View::BorrowMut(v) => View::BorrowMut(v)
        };

        Mat {
            nrows: row_range.end - row_range.start,
            ncols: col_range.end - col_range.start,
            offset: self.offset + self.stride * col_range.start + row_range.start,
            view,
            .. *self
        }
    }
    //
    pub fn row(&self, r: usize) -> Mat
    {
        self.slice(r ..= r, ..)
    }
    //
    pub fn col(&self, c: usize) -> Mat
    {
        self.slice(.., c ..= c)
    }
    //
    pub fn row_mut(&mut self, r: usize) -> Mat
    {
        self.slice_mut(r ..= r, ..)
    }
    //
    pub fn col_mut(&mut self, c: usize) -> Mat
    {
        self.slice_mut(.., c ..= c)
    }
    //
    fn tr_index(&self, index: (usize, usize)) -> usize
    {
        if !self.transposed {
            self.offset + self.stride * index.1 + index.0
        }
        else {
            self.offset + self.stride * index.0 + index.1
        }
    }
    //
    pub fn set_by<F>(mut self, f: F) -> Mat<'a>
    where F: Fn((usize, usize)) -> FP
    {
        for c in 0 .. self.ncols {
            for r in 0 .. self.nrows {
                self[(r, c)] = f((r, c));
            }
        }
        self
    }
    //
    pub fn set_eye(mut self) -> Mat<'a>
    {
        for c in 0 .. self.ncols {
            for r in 0 .. self.nrows {
                self[(r, c)] = if r == c {1.} else {0.};
            }
        }
        self
    }
    //
    pub fn set_iter<'b, T>(mut self, iter: T) -> Mat<'a>
    where T: IntoIterator<Item=&'b FP>
    {
        // NOTE: read row-wise
        let mut i = iter.into_iter();
        for r in 0 .. self.nrows {
            for c in 0 .. self.ncols {
                self[(r, c)] = *i.next().unwrap_or(&0.);
            }
        }
        self
    }
    //
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
    pub fn assign(&mut self, rhs: &Mat)
    {
        let (l_nrows, l_ncols) = self.size();
        let (r_nrows, r_ncols) = rhs.size();

        assert_eq!(l_nrows, r_nrows);
        assert_eq!(l_ncols, r_ncols);
        
        for r in 0 .. self.nrows {
            for c in 0 .. self.ncols {
                self[(r, c)] = rhs[(r, c)];
            }
        }
    }
    //
    pub fn t(&self) -> Mat
    {
        let view = match &self.view {
            View::Own(v) => View::Borrow(&v),
            View::Borrow(v) => View::Borrow(&v),
            View::BorrowMut(v) => View::Borrow(&v)
        };

        Mat {
            transposed: !self.transposed,
            view,
            .. *self
        }
    }
    //
    pub fn diag(&self) -> Mat
    {
        let (l_nrows, _) = self.size();

        let mut mat = Mat::new(l_nrows, l_nrows);

        for r in 0 .. l_nrows {
            mat[(r, r)] = self[(r, 0)];
        }

        mat
    }
    //
    pub fn sq_sum(&self) -> FP
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
}

//

impl<'a> Index<(usize, usize)> for Mat<'a>
{
    type Output = FP;
    fn index(&self, index: (usize, usize)) -> &FP
    {
        let i = self.tr_index(index);

        match &self.view {
            View::Own(v) => &v[i],
            View::Borrow(v) => &v[i],
            View::BorrowMut(v) => &v[i]
        }
    }
}

impl<'a> IndexMut<(usize, usize)> for Mat<'a>
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut FP
    {
        let i = self.tr_index(index);

        match &mut self.view {
            View::Own(v) => &mut v[i],
            View::Borrow(_) => panic!("cannot index Borrow as mutable"),
            View::BorrowMut(v) => &mut v[i]
        }
    }
}

//

impl<'a> fmt::Display for Mat<'a>
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

impl<'a> PartialEq for Mat<'a>
{
    fn eq(&self, other: &Self) -> bool
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
    fn size(&self) -> (usize, usize);
    fn get(&self, row: usize, col: usize) -> FP;
}

impl<'a> MatAcc for Mat<'a>
{
    fn size(&self) -> (usize, usize)
    {
        self.size()
    }
    //
    fn get(&self, row: usize, col: usize) -> FP
    {
        self[(row, col)]
    }
}

impl<'a> MatAcc for &Mat<'a>
{
    fn size(&self) -> (usize, usize)
    {
        (*self).size()
    }
    //
    fn get(&self, row: usize, col: usize) -> FP
    {
        (*self).get(row, col)
    }
}

//

impl<'a> Neg for &Mat<'a>
{
    type Output = Mat<'static>;

    fn neg(self) -> Mat<'static>
    {
        let (l_nrows, l_ncols) = self.size();

        let mut mat = Mat::new(l_nrows, l_ncols);

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                mat[(r, c)] = -self.get(r, c);
            }
        }

        mat
    }
}

impl<'a> Neg for Mat<'a>
{
    type Output = Mat<'static>;

    fn neg(self) -> Mat<'static>
    {
        -&self
    }
}

//

impl<'a, T> Add<T> for &Mat<'a>
where T: MatAcc
{
    type Output = Mat<'static>;

    fn add(self, rhs: T) -> Mat<'static>
    {
        let (l_nrows, l_ncols) = self.size();

        assert_eq!((l_nrows, l_ncols), rhs.size());

        let mut mat = Mat::new(l_nrows, l_ncols);

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                mat[(r, c)] = self.get(r, c) + rhs.get(r, c);
            }
        }

        mat
    }
}

impl<'a> Add<FP> for &Mat<'a>
{
    type Output = Mat<'static>;

    fn add(self, rhs: FP) -> Mat<'static>
    {
        let (l_nrows, l_ncols) = self.size();

        let mut mat = Mat::new(l_nrows, l_ncols);

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                mat[(r, c)] = self.get(r, c) + rhs;
            }
        }

        mat
    }
}

impl<'a, T> Add<T> for Mat<'a>
where T: MatAcc
{
    type Output = Mat<'static>;

    fn add(self, rhs: T) -> Mat<'static>
    {
        &self + rhs
    }
}

impl<'a> Add<FP> for Mat<'a>
{
    type Output = Mat<'static>;

    fn add(self, rhs: FP) -> Mat<'static>
    {
        &self + rhs
    }
}

impl<'a> Add<&Mat<'a>> for FP
{
    type Output = Mat<'static>;

    fn add(self, rhs: &Mat) -> Mat<'static>
    {
        rhs.add(self)
    }
}

impl<'a> Add<Mat<'a>> for FP
{
    type Output = Mat<'static>;

    fn add(self, rhs: Mat) -> Mat<'static>
    {
        rhs.add(self)
    }
}

//

impl<'a, T> Sub<T> for &Mat<'a>
where T: MatAcc
{
    type Output = Mat<'static>;

    fn sub(self, rhs: T) -> Mat<'static>
    {
        let (l_nrows, l_ncols) = self.size();

        assert_eq!((l_nrows, l_ncols), rhs.size());

        let mut mat = Mat::new(l_nrows, l_ncols);

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                mat[(r, c)] = self.get(r, c) - rhs.get(r, c);
            }
        }

        mat
    }
}

impl<'a> Sub<FP> for &Mat<'a>
{
    type Output = Mat<'static>;

    fn sub(self, rhs: FP) -> Mat<'static>
    {
        let (l_nrows, l_ncols) = self.size();

        let mut mat = Mat::new(l_nrows, l_ncols);

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                mat[(r, c)] = self.get(r, c) - rhs;
            }
        }

        mat
    }
}

impl<'a, T> Sub<T> for Mat<'a>
where T: MatAcc
{
    type Output = Mat<'static>;

    fn sub(self, rhs: T) -> Mat<'static>
    {
        &self - rhs
    }
}

impl<'a> Sub<FP> for Mat<'a>
{
    type Output = Mat<'static>;

    fn sub(self, rhs: FP) -> Mat<'static>
    {
        &self - rhs
    }
}

impl<'a> Sub<&Mat<'a>> for FP
{
    type Output = Mat<'static>;

    fn sub(self, rhs: &Mat) -> Mat<'static>
    {
        -rhs + self
    }
}

impl<'a> Sub<Mat<'a>> for FP
{
    type Output = Mat<'static>;

    fn sub(self, rhs: Mat) -> Mat<'static>
    {
        -rhs + self
    }
}

//

impl<'a, T> Mul<T> for &Mat<'a>
where T: MatAcc
{
    type Output = Mat<'static>;

    fn mul(self, rhs: T) -> Mat<'static>
    {
        let (l_nrows, l_ncols) = self.size();
        let (r_nrows, r_ncols) = rhs.size();

        assert_eq!(l_ncols, r_nrows);

        let mut mat = Mat::new(l_nrows, r_ncols);

        for c in 0 .. r_ncols {
            for r in 0 .. l_nrows {
                let mut v: FP = 0.0;
                for k in 0 .. l_ncols {
                    v += self.get(r, k) * rhs.get(k, c);
                }
                mat[(r, c)] = v;
            }
        }

        mat
    }
}

impl<'a> Mul<FP> for &Mat<'a>
{
    type Output = Mat<'static>;

    fn mul(self, rhs: FP) -> Mat<'static>
    {
        let (l_nrows, l_ncols) = self.size();

        let mut mat = Mat::new(l_nrows, l_ncols);

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                mat[(r, c)] = self.get(r, c) * rhs;
            }
        }

        mat
    }
}

impl<'a, T> Mul<T> for Mat<'a>
where T: MatAcc
{
    type Output = Mat<'static>;

    fn mul(self, rhs: T) -> Mat<'static>
    {
        &self * rhs
    }
}

impl<'a> Mul<FP> for Mat<'a>
{
    type Output = Mat<'static>;

    fn mul(self, rhs: FP) -> Mat<'static>
    {
        &self * rhs
    }
}

impl<'a> Mul<&Mat<'a>> for FP
{
    type Output = Mat<'static>;

    fn mul(self, rhs: &Mat) -> Mat<'static>
    {
        rhs.mul(self)
    }
}

impl<'a> Mul<Mat<'a>> for FP
{
    type Output = Mat<'static>;

    fn mul(self, rhs: Mat) -> Mat<'static>
    {
        rhs.mul(self)
    }
}

//

impl<'a, T> Div<T> for &Mat<'a>
where T: MatAcc
{
    type Output = Mat<'static>;

    fn div(self, _rhs: T) -> Mat<'static>
    {
        panic!("not supported");
    }
}

impl<'a> Div<FP> for &Mat<'a>
{
    type Output = Mat<'static>;

    fn div(self, rhs: FP) -> Mat<'static>
    {
        let (l_nrows, l_ncols) = self.size();

        let mut mat = Mat::new(l_nrows, l_ncols);

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                mat[(r, c)] = self.get(r, c) / rhs;
            }
        }

        mat
    }
}

impl<'a, T> Div<T> for Mat<'a>
where T: MatAcc
{
    type Output = Mat<'static>;

    fn div(self, _rhs: T) -> Mat<'static>
    {
        panic!("not supported");
    }
}

impl<'a> Div<FP> for Mat<'a>
{
    type Output = Mat<'static>;

    fn div(self, rhs: FP) -> Mat<'static>
    {
        &self / rhs
    }
}

impl<'a> Div<&Mat<'a>> for FP
{
    type Output = Mat<'static>;

    fn div(self, _rhs: &Mat) -> Mat<'static>
    {
        panic!("not supported");
    }
}

impl<'a> Div<Mat<'a>> for FP
{
    type Output = Mat<'static>;

    fn div(self, _rhs: Mat) -> Mat<'static>
    {
        panic!("not supported");
    }
}

//

#[test]
fn test_set()
{
    {
        let a = Mat::new(3, 3).set_eye();
        let b = Mat::new(3, 3).set_iter(&[
            1., 0., 0.,
            0., 1., 0.,
            0., 0., 1.
        ]);
        assert_eq!(a, b);
    }
    {
        let a = Mat::new(2, 4).set_by(|(r, c)| {(r * 4 + c) as FP});
        let b = Mat::new(2, 4).set_iter(&[
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
        let a = Mat::new1(3);
        let a = a.t();
        let b = Mat::new(1, 3);
        assert_eq!(a, b);
    }
    {
        let mut a = Mat::new(4, 4);
        let b = Mat::new(4, 4).set_by(|_| {rand::random()});
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
        let a2 = Mat::new(2, 2).set_by(|_| {2.0});
        a1.assign(&a2);
        assert_eq!(a, b);
    }
}

// TODO: test
#[test]
fn test()
{
    {
        let a = Mat::new(3, 3).set_eye();
        let b = Mat::new(3, 3).set_iter(&[
            1., 0., 0.,
            0., 1., 0.,
            0., 0., 1.
        ]);
        let c = &a + (&b + &a);
        println!("{}", c);
        let c = &a + 1.;
        println!("{:?}", c);
        let c = 1. + &b;
        println!("{:?}", c);
        println!("{:?}", a);
        println!("{:?}", b);
        println!();
    }
    {
        let mut a = Mat::new(3, 3).set_eye();
        let b = Mat::new(3, 3).set_iter(&[
            1., 0., 0.,
            0., 1., 0.,
            0., 0., 1.
        ]);
        let c = &a + &b;
        a.assign(&c);
    }
}
