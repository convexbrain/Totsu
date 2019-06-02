/*!
Matrix

[`Mat`](type.Mat.html) is primal type used to make and own matrix value.
See also [`MatGen`](struct.MatGen.html) for supported methods.
*/

use std::cmp::PartialEq;
use std::ops::{Range, RangeBounds, Bound};
use std::ops::{Neg, Add, Mul, Sub, Div, AddAssign, SubAssign, MulAssign, DivAssign};
use std::ops::{Index, IndexMut};
use std::fmt;

/// Scalar floating point type
pub type FP = f64;
pub use std::f64::EPSILON as FP_EPSILON;
pub use std::f64::MIN_POSITIVE as FP_MINPOS;

/// Matrix
pub type Mat = MatGen<Vec<FP>>;
/// Matrix slice
pub type MatSlice<'a> = MatGen<&'a Vec<FP>>;
/// Matrix slice mutable
pub type MatSliMu<'a> = MatGen<&'a mut Vec<FP>>;

impl MatView for Vec<FP>
{
    type OwnColl = Vec<FP>;

    fn new_own(sz: usize) -> Self::OwnColl
    {
        vec![0.; sz]
    }
    fn get_ref(&self) -> &Self::OwnColl
    {
        self
    }
    fn get_mut(&mut self) -> &mut Self::OwnColl
    {
        self
    }
    fn get_len(&self) -> usize
    {
        self.len()
    }
    fn get_own(self) -> Self::OwnColl
    {
        self
    }
    fn is_own(&self) -> bool
    {
        true
    }
    fn clone_own(&self) -> Self::OwnColl
    {
        self.clone()
    }
    fn get_index(&self, i: usize) -> &FP
    {
        &self[i]
    }
    fn get_index_mut(&mut self, i: usize) -> &mut FP
    {
        &mut self[i]
    }
    fn get_iter_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item=(usize, &mut FP)> + 'a> {
        Box::new(self.iter_mut().enumerate())
    }
}

impl MatView for &Vec<FP>
{
    type OwnColl = Vec<FP>;

    fn new_own(sz: usize) -> Self::OwnColl
    {
        vec![0.; sz]
    }
    fn get_ref(&self) -> &Self::OwnColl
    {
        self
    }
    fn get_mut(&mut self) -> &mut Self::OwnColl
    {
        panic!("cannot borrow immutable as mutable");
    }
    fn get_len(&self) -> usize
    {
        self.len()
    }
    fn get_own(self) -> Self::OwnColl
    {
        panic!("cannot own immutable");
    }
    fn is_own(&self) -> bool
    {
        false
    }
    fn clone_own(&self) -> Self::OwnColl
    {
        (*self).clone()
    }
    fn get_index(&self, i: usize) -> &FP
    {
        &self[i]
    }
    fn get_index_mut(&mut self, _i: usize) -> &mut FP
    {
        panic!("cannot borrow immutable as mutable");
    }
    fn get_iter_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item=(usize, &mut FP)> + 'a> {
        panic!("cannot borrow immutable as mutable");
    }
}

impl MatView for &mut Vec<FP>
{
    type OwnColl = Vec<FP>;

    fn new_own(sz: usize) -> Self::OwnColl
    {
        vec![0.; sz]
    }
    fn get_ref(&self) -> &Self::OwnColl
    {
        self
    }
    fn get_mut(&mut self) -> &mut Self::OwnColl
    {
        self
    }
    fn get_len(&self) -> usize
    {
        self.len()
    }
    fn get_own(self) -> Self::OwnColl
    {
        panic!("cannot own mutable");
    }
    fn is_own(&self) -> bool
    {
        false
    }
    fn clone_own(&self) -> Self::OwnColl
    {
        (*self).clone()
    }
    fn get_index(&self, i: usize) -> &FP
    {
        &self[i]
    }
    fn get_index_mut(&mut self, i: usize) -> &mut FP
    {
        &mut self[i]
    }
    fn get_iter_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item=(usize, &mut FP)> + 'a> {
        Box::new(self.iter_mut().enumerate())
    }
}

impl Clone for Mat
{
    fn clone(&self) -> Mat
    {
        self.clone_sz()
    }
}

//

/// Ownership view of matrix array entity
pub trait MatView {
    type OwnColl: MatView;

    fn new_own(sz: usize) -> Self::OwnColl;
    fn get_ref(&self) -> &Self::OwnColl;
    fn get_mut(&mut self) -> &mut Self::OwnColl;
    fn get_len(&self) -> usize;
    fn get_own(self) -> Self::OwnColl;
    fn is_own(&self) -> bool;
    fn clone_own(&self) -> Self::OwnColl;
    fn get_index(&self, i: usize) -> &FP;
    fn get_index_mut(&mut self, i: usize) -> &mut FP;
    fn put(&mut self, i: usize, val: FP) {
        *self.get_index_mut(i) = val;
    }
    fn is_less(&self, _sz: (usize, usize)) -> bool {
        true // TODO
    }
    fn get_iter_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item=(usize, &mut FP)> + 'a>;
}

/// Generic struct of matrix
#[derive(Debug)]
pub struct MatGen<V: MatView>
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

impl<V: MatView> MatGen<V>
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
    fn h_own(self) -> MatGen<V::OwnColl>
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
            self.clone_sz()
        }
    }
    //
    /// *new* - Makes a matrix.
    pub fn new(nrows: usize, ncols: usize) -> MatGen<V::OwnColl>
    {
        MatGen {
            nrows,
            ncols,
            offset: 0,
            stride: nrows,
            transposed: false,
            view: V::new_own(nrows * ncols)
        }
    }
    /// *new* - Makes a matrix of the same size.
    pub fn new_like<V2: MatView>(mat: &MatGen<V2>) -> MatGen<V::OwnColl>
    {
        let (nrows, ncols) = mat.size();

        MatGen::<V>::new(nrows, ncols)
    }
    /// *new* - Makes a column vector.
    pub fn new_vec(nrows: usize) -> MatGen<V::OwnColl>
    {
        MatGen::<V>::new(nrows, 1)
    }
    //
    /// *slice* - Slice block reference.
    pub fn slice<'a, RR, CR>(&'a self, rows: RR, cols: CR) -> MatGen<&V::OwnColl>
    where RR: RangeBounds<usize>,  CR: RangeBounds<usize>, &'a V::OwnColl: MatView
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
    /// *slice* - Slice block mutable reference.
    pub fn slice_mut<'a, RR, CR>(&'a mut self, rows: RR, cols: CR) -> MatGen<&mut V::OwnColl>
    where RR: RangeBounds<usize>,  CR: RangeBounds<usize>, &'a mut V::OwnColl: MatView
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
    /// *slice* - Row vectors reference.
    pub fn rows<'a, RR>(&'a self, rows: RR) -> MatGen<&V::OwnColl>
    where RR: RangeBounds<usize>, &'a V::OwnColl: MatView
    {
        self.slice(rows, ..)
    }
    /// *slice* - Column vectors reference.
    pub fn cols<'a, CR>(&'a self, cols: CR) -> MatGen<&V::OwnColl>
    where CR: RangeBounds<usize>, &'a V::OwnColl: MatView
    {
        self.slice(.., cols)
    }
    /// *slice* - A row vector reference.
    pub fn row<'a>(&'a self, r: usize) -> MatGen<&V::OwnColl>
    where &'a V::OwnColl: MatView
    {
        self.rows(r ..= r)
    }
    /// *slice* - A column vector reference.
    pub fn col<'a>(&'a self, c: usize) -> MatGen<&V::OwnColl>
    where &'a V::OwnColl: MatView
    {
        self.cols(c ..= c)
    }
    /// *slice* - Row vectors mutable reference.
    pub fn rows_mut<'a, RR>(&'a mut self, rows: RR) -> MatGen<&mut V::OwnColl>
    where RR: RangeBounds<usize>, &'a mut V::OwnColl: MatView
    {
        self.slice_mut(rows, ..)
    }
    /// *slice* - Column vectors mutable reference.
    pub fn cols_mut<'a, CR>(&'a mut self, cols: CR) -> MatGen<&mut V::OwnColl>
    where CR: RangeBounds<usize>, &'a mut V::OwnColl: MatView
    {
        self.slice_mut(.., cols)
    }
    /// *slice* - A row vector mutable reference.
    pub fn row_mut<'a>(&'a mut self, r: usize) -> MatGen<&mut V::OwnColl>
    where &'a mut V::OwnColl: MatView
    {
        self.rows_mut(r ..= r)
    }
    /// *slice* - A column vector mutable reference.
    pub fn col_mut<'a>(&'a mut self, c: usize) -> MatGen<&mut V::OwnColl>
    where &'a mut V::OwnColl: MatView
    {
        self.cols_mut(c ..= c)
    }
    /// *slice* - Whole reference.
    pub fn as_slice<'a>(&'a self) -> MatGen<&V::OwnColl>
    where &'a V::OwnColl: MatView
    {
        self.slice(.., ..)
    }
    /// *slice* - Whole mutable reference.
    pub fn as_slice_mut<'a>(&'a mut self) -> MatGen<&mut V::OwnColl>
    where &'a mut V::OwnColl: MatView
    {
        self.slice_mut(.., ..)
    }
    /// *slice* - Transopsed reference.
    pub fn t<'a>(&'a self) -> MatGen<&V::OwnColl>
    where &'a V::OwnColl: MatView
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
    /// *slice* - Transopsed mutable reference.
    pub fn t_mut<'a>(&'a mut self) -> MatGen<&mut V::OwnColl>
    where &'a mut V::OwnColl: MatView
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
    /// *set* - Set by closure.
    pub fn set_by<F>(mut self, mut f: F) -> MatGen<V>
    where F: FnMut(usize, usize) -> FP
    {
        self.assign_by(|r, c| Some(f(r, c)));
        self
    }
    /// *set* - Set by iterator.
    pub fn set_iter<'b, T>(mut self, iter: T) -> MatGen<V>
    where T: IntoIterator<Item=&'b FP>
    {
        self.assign_iter(iter);
        self
    }
    /// *set* - Set eye matrix with a value.
    pub fn set_eye(mut self, value: FP) -> MatGen<V>
    {
        self.assign_eye(value);
        self
    }
    /// *set* - Set a value.
    pub fn set_all(mut self, value: FP) -> MatGen<V>
    {
        self.assign_all(value);
        self
    }
    /// *set* - Set transposed.
    pub fn set_t(mut self) -> MatGen<V>
    {
        self.transposed = !self.transposed;
        self
    }
    //
    /// *clone* - Clone with shrinking size.
    pub fn clone_sz(&self) -> MatGen<V::OwnColl>
    {
        let (l_nrows, l_ncols) = self.size();
        let sz = self.view.get_len();

        if sz == l_nrows * l_ncols {
            MatGen {
                nrows: self.nrows,
                ncols: self.ncols,
                offset: self.offset,
                stride: self.stride,
                transposed: self.transposed,
                view: self.view.clone_own()
            }
        }
        else {
            let mut mat = MatGen::<V>::new(l_nrows, l_ncols);
            mat.assign(self);
            mat
        }
    }
    /// *clone* - Clone into diagonal matrix.
    pub fn clone_diag(&self) -> MatGen<V::OwnColl>
    {
        let (l_nrows, l_ncols) = self.size();
        assert_eq!(l_ncols, 1);

        let mut mat = MatGen::<V>::new(l_nrows, l_nrows);

        for r in 0 .. l_nrows {
            mat.a((r, r), self[(r, 0)]);
        }

        mat
    }
    //
    /// *assign* - Assign by index
    pub fn a(&mut self, index: (usize, usize), val: FP)
    {
        let i = self.h_index(index);

        self.view.put(i, val);
    }
    //
    /// *assign* - Assign by closure.
    pub fn assign_by<F>(&mut self, mut f: F)
    where F: FnMut(usize, usize) -> Option<FP>
    {
        let (l_nrows, l_ncols) = self.size();

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                if let Some(value) = f(r, c) {
                    self.a((r, c), value);
                }
            }
        }
    }
    /// *assign* - Assign by iterator.
    pub fn assign_iter<'b, T>(&mut self, iter: T)
    where T: IntoIterator<Item=&'b FP>
    {
        let (nrows, ncols) = self.size();
        let mut i = iter.into_iter();

        // NOTE: contents of iter is row-wise
        for r in 0 .. nrows {
            for c in 0 .. ncols {
                self.a((r, c), *i.next().unwrap_or(&0.));
            }
        }
    }
    /// *assign* - Assign eye matrix with a value.
    pub fn assign_eye(&mut self, value: FP)
    {
        self.assign_by(|r, c| Some(if r == c {value} else {0.}));
    }
    /// *assign* - Assign a value.
    pub fn assign_all(&mut self, value: FP)
    {
        self.assign_by(|_, _| Some(value));
    }
    /// *assign* - Assign matrix with scaling.
    pub fn assign_s<V2: MatView>(&mut self, rhs: &MatGen<V2>, value: FP)
    {
        let (l_nrows, l_ncols) = self.size();
        let (r_nrows, r_ncols) = rhs.size();

        assert_eq!(l_nrows, r_nrows);
        assert_eq!(l_ncols, r_ncols);
        
        self.assign_by(|r, c| Some(value * rhs[(r, c)]));
    }
    /// *assign* - Assign matrix.
    pub fn assign<V2: MatView>(&mut self, rhs: &MatGen<V2>)
    {
        self.assign_s(rhs, 1.);
    }
    //
    /// Returns p=2 norm squared.
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
    /// Returns p=2 norm.
    pub fn norm_p2(&self) -> FP
    {
        FP::sqrt(self.norm_p2sq())
    }
    /// Returns trace.
    pub fn tr(&self) -> FP
    {
        let (l_nrows, l_ncols) = self.size();

        let mut sum = 0.;

        for i in 0 .. l_nrows.min(l_ncols) {
            sum += self[(i, i)];
        }

        sum
    }
    /// Returns inner product.
    pub fn prod<V2: MatView>(&self, rhs: &MatGen<V2>) -> FP
    {
        let (l_nrows, l_ncols) = self.size();
        let (r_nrows, r_ncols) = rhs.size();

        assert_eq!(l_nrows, r_nrows);
        assert_eq!(l_ncols, r_ncols);

        let mut sum = 0.;

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                sum += self[(r, c)] * rhs[(r, c)];
            }
        }

        sum
    }
    //
    /// Finds maximum value.
    pub fn max(&self) -> Option<FP>
    {
        let (l_nrows, l_ncols) = self.size();
        if (l_nrows == 0) || (l_ncols == 0) {
            return None;
        }

        let mut m = self[(0, 0)];

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                if self[(r, c)] > m {
                    m = self[(r, c)];
                }
            }
        }

        Some(m)
    }
    /// Finds minumum value.
    pub fn min(&self) -> Option<FP>
    {
        let (l_nrows, l_ncols) = self.size();
        if (l_nrows == 0) || (l_ncols == 0) {
            return None;
        }
        
        let mut m = self[(0, 0)];

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                if self[(r, c)] < m {
                    m = self[(r, c)];
                }
            }
        }

        Some(m)
    }
    //
    /// Returns number of rows and columns.
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
    /// Made into diagonal matrix and multiplied.
    pub fn diag_mul<'a, V2: MatView>(&self, rhs: &'a MatGen<V2>) -> MatGen<V::OwnColl>
    where &'a V2::OwnColl: MatView
    {
        let (l_nrows, l_ncols) = self.size();
        let (r_nrows, r_ncols) = rhs.size();

        assert_eq!(l_ncols, 1);
        assert_eq!(l_nrows, r_nrows);

        let mut mat = MatGen::<V>::new_like(rhs);

        for c in 0 .. r_ncols {
            for r in 0 .. r_nrows {
                mat.a((r, c), self[(r, 0)] * rhs[(r, c)]);
            }
        }

        mat
    }
    //
    fn ops_neg(&mut self)
    {
        if self.view.is_less(self.size()) {
            for (_, _, val) in self.iter_mut() {
                *val = -(*val);
            }
        }
        else {
            let (l_nrows, l_ncols) = self.size();

            for c in 0 .. l_ncols {
                for r in 0 .. l_nrows {
                    self.a((r, c), -self[(r, c)]);
                }
            }
        }
    }
    //
    fn iter_mut(&mut self) -> MatIterMut
    {
        MatIterMut {
            p: MatProp {
                nrows: self.nrows,
                ncols: self.ncols,
                offset: self.offset,
                stride: self.stride,
                transposed: self.transposed
            },
            iter: self.view.get_iter_mut()
        }
    }
}

// TODO

#[derive(Clone)]
struct MatProp
{
    nrows: usize,
    ncols: usize,
    //
    offset: usize,
    stride: usize,
    //
    transposed: bool
}

struct MatIterMut<'a>
{
    p: MatProp,
    //
    iter: Box<dyn Iterator<Item=(usize, &'a mut FP)> + 'a>
}

impl MatProp
{
    fn in_index(&self, i: usize) -> Option<(usize, usize)>
    {
        if i >= self.offset {
            let i = i - self.offset;

            let i0 = i % self.stride;
            let i1 = i / self.stride;

            if !self.transposed {
                if i0 < self.nrows && i1 < self.ncols {
                    return Some((i0, i1));
                }
            }
            else {
                if i1 < self.nrows && i0 < self.ncols {
                    return Some((i1, i0));
                }
            }
        }

        None
    }
}

impl<'a> Iterator for MatIterMut<'a>
{
    type Item = (usize, usize, &'a mut FP);

    fn next(&mut self) -> Option<Self::Item>
    {
        while let Some((idx, val)) = self.iter.next() {
            if let Some((r, c)) = self.p.in_index(idx) {
                return Some((r, c, val));
            }
        }

        None
    }
}

//

impl<V: MatView> Index<(usize, usize)> for MatGen<V>
{
    type Output = FP;
    fn index(&self, index: (usize, usize)) -> &FP
    {
        let i = self.h_index(index);

        self.view.get_index(i)
    }
}

impl<V: MatView> IndexMut<(usize, usize)> for MatGen<V>
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut FP
    {
        let i = self.h_index(index);

        self.view.get_index_mut(i)
    }
}

//

impl<V: MatView> fmt::LowerExp for MatGen<V>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error>
    {
        let (l_nrows, l_ncols) = self.size();

        writeln!(f, "[")?;
        for r in 0 .. l_nrows {
            for c in 0 .. l_ncols {
                write!(f, "  {:.precision$e},", self[(r, c)], precision = f.precision().unwrap_or(3))?;
            }
            writeln!(f)?;
        }
        write!(f, "]")
    }
}

impl<V: MatView> fmt::Display for MatGen<V>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error>
    {
        writeln!(f, "{:.3e}", self)
    }
}

//

impl<V: MatView, V2: MatView> PartialEq<MatGen<V2>> for MatGen<V>
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

/// Helper matrix accessor for operator overload
pub trait MatAcc
{
    fn acc_size(&self) -> (usize, usize);
    fn acc_get(&self, row: usize, col: usize) -> FP;
}

impl<V: MatView> MatAcc for MatGen<V>
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

impl<V: MatView> MatAcc for &MatGen<V>
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
impl<V: MatView> Neg for MatGen<V>
{
    type Output = MatGen<V::OwnColl>;

    fn neg(self) -> MatGen<V::OwnColl>
    {
        let mut mat = self.h_own();
        mat.ops_neg();
        mat
    }
}

impl<V: MatView> Neg for &MatGen<V>
{
    type Output = MatGen<V::OwnColl>;

    fn neg(self) -> MatGen<V::OwnColl>
    {
        let mut mat = self.clone_sz();
        mat.ops_neg();
        mat
    }
}

//
impl<V: MatView, T: MatAcc> AddAssign<T> for MatGen<V>
{
    fn add_assign(&mut self, rhs: T)
    {
        let (l_nrows, l_ncols) = self.size();

        assert_eq!((l_nrows, l_ncols), rhs.acc_size());

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                self.a((r, c), self[(r, c)] + rhs.acc_get(r, c));
            }
        }
    }
}

impl<V: MatView> AddAssign<FP> for MatGen<V>
{
    fn add_assign(&mut self, rhs: FP)
    {
        let (l_nrows, l_ncols) = self.size();

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                self.a((r, c), self[(r, c)] + rhs);
            }
        }
    }
}

impl<V: MatView, T: MatAcc> Add<T> for MatGen<V>
{
    type Output = MatGen<V::OwnColl>;

    fn add(self, rhs: T) -> MatGen<V::OwnColl>
    {
        let mut mat = self.h_own();
        mat.add_assign(rhs);
        mat
    }
}

impl<V: MatView, T: MatAcc> Add<T> for &MatGen<V>
{
    type Output = MatGen<V::OwnColl>;

    fn add(self, rhs: T) -> MatGen<V::OwnColl>
    {
        let mut mat = self.clone_sz();
        mat.add_assign(rhs);
        mat
    }
}

impl<V: MatView> Add<FP> for MatGen<V>
{
    type Output = MatGen<V::OwnColl>;

    fn add(self, rhs: FP) -> MatGen<V::OwnColl>
    {
        let mut mat = self.h_own();
        mat.add_assign(rhs);
        mat
    }
}

impl<V: MatView> Add<FP> for &MatGen<V>
{
    type Output = MatGen<V::OwnColl>;

    fn add(self, rhs: FP) -> MatGen<V::OwnColl>
    {
        let mut mat = self.clone_sz();
        mat.add_assign(rhs);
        mat
    }
}

impl<V: MatView> Add<MatGen<V>> for FP
{
    type Output = MatGen<V::OwnColl>;

    fn add(self, rhs: MatGen<V>) -> MatGen<V::OwnColl>
    {
        rhs.add(self)
    }
}

impl<V: MatView> Add<&MatGen<V>> for FP
{
    type Output = MatGen<V::OwnColl>;

    fn add(self, rhs: &MatGen<V>) -> MatGen<V::OwnColl>
    {
        rhs.add(self)
    }
}

//

impl<V: MatView, T: MatAcc> SubAssign<T> for MatGen<V>
{
    fn sub_assign(&mut self, rhs: T)
    {
        let (l_nrows, l_ncols) = self.size();

        assert_eq!((l_nrows, l_ncols), rhs.acc_size());

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                self.a((r, c), self[(r, c)] - rhs.acc_get(r, c));
            }
        }
    }
}

impl<V: MatView> SubAssign<FP> for MatGen<V>
{
    fn sub_assign(&mut self, rhs: FP)
    {
        let (l_nrows, l_ncols) = self.size();

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                self.a((r, c), self[(r, c)] - rhs);
            }
        }
    }
}

impl<V: MatView, T: MatAcc> Sub<T> for MatGen<V>
{
    type Output = MatGen<V::OwnColl>;

    fn sub(self, rhs: T) -> MatGen<V::OwnColl>
    {
        let mut mat = self.h_own();
        mat.sub_assign(rhs);
        mat
    }
}

impl<V: MatView, T: MatAcc> Sub<T> for &MatGen<V>
{
    type Output = MatGen<V::OwnColl>;

    fn sub(self, rhs: T) -> MatGen<V::OwnColl>
    {
        let mut mat = self.clone_sz();
        mat.sub_assign(rhs);
        mat
    }
}

impl<V: MatView> Sub<FP> for MatGen<V>
{
    type Output = MatGen<V::OwnColl>;

    fn sub(self, rhs: FP) -> MatGen<V::OwnColl>
    {
        let mut mat = self.h_own();
        mat.sub_assign(rhs);
        mat
    }
}

impl<V: MatView> Sub<FP> for &MatGen<V>
{
    type Output = MatGen<V::OwnColl>;

    fn sub(self, rhs: FP) -> MatGen<V::OwnColl>
    {
        let mut mat = self.clone_sz();
        mat.sub_assign(rhs);
        mat
    }
}

impl<V: MatView> Sub<MatGen<V>> for FP
{
    type Output = MatGen<V::OwnColl>;

    fn sub(self, rhs: MatGen<V>) -> MatGen<V::OwnColl>
    {
        let mut mat = rhs.neg();
        mat.add_assign(self);
        mat
    }
}

impl<V: MatView> Sub<&MatGen<V>> for FP
{
    type Output = MatGen<V::OwnColl>;

    fn sub(self, rhs: &MatGen<V>) -> MatGen<V::OwnColl>
    {
        let mut mat = rhs.neg();
        mat.add_assign(self);
        mat
    }
}

//

impl<V: MatView> MulAssign<FP> for MatGen<V>
{
    fn mul_assign(&mut self, rhs: FP)
    {
        let (l_nrows, l_ncols) = self.size();

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                self.a((r, c), self[(r, c)] * rhs);
            }
        }
    }
}

impl<V: MatView, T: MatAcc> Mul<T> for MatGen<V>
{
    type Output = MatGen<V::OwnColl>;

    fn mul(self, rhs: T) -> MatGen<V::OwnColl>
    {
        (&self).mul(rhs)
    }
}

impl<V: MatView, T: MatAcc> Mul<T> for &MatGen<V>
{
    type Output = MatGen<V::OwnColl>;

    fn mul(self, rhs: T) -> MatGen<V::OwnColl>
    {
        let (l_nrows, l_ncols) = self.size();
        let (r_nrows, r_ncols) = rhs.acc_size();

        assert_eq!(l_ncols, r_nrows);

        let mut mat = MatGen::<V>::new(l_nrows, r_ncols);

        for c in 0 .. r_ncols {
            for r in 0 .. l_nrows {
                let mut v: FP = 0.0;
                for k in 0 .. l_ncols {
                    v += self[(r, k)] * rhs.acc_get(k, c);
                }
                mat.a((r, c), v);
            }
        }

        mat
    }
}

impl<V: MatView> Mul<FP> for MatGen<V>
{
    type Output = MatGen<V::OwnColl>;

    fn mul(self, rhs: FP) -> MatGen<V::OwnColl>
    {
        let mut mat = self.h_own();
        mat.mul_assign(rhs);
        mat
    }
}

impl<V: MatView> Mul<FP> for &MatGen<V>
{
    type Output = MatGen<V::OwnColl>;

    fn mul(self, rhs: FP) -> MatGen<V::OwnColl>
    {
        let mut mat = self.clone_sz();
        mat.mul_assign(rhs);
        mat
    }
}

impl<V: MatView> Mul<MatGen<V>> for FP
{
    type Output = MatGen<V::OwnColl>;

    fn mul(self, rhs: MatGen<V>) -> MatGen<V::OwnColl>
    {
        rhs.mul(self)
    }
}

impl<V: MatView> Mul<&MatGen<V>> for FP
{
    type Output = MatGen<V::OwnColl>;

    fn mul(self, rhs: &MatGen<V>) -> MatGen<V::OwnColl>
    {
        rhs.mul(self)
    }
}

//

impl<V: MatView> DivAssign<FP> for MatGen<V>
{
    fn div_assign(&mut self, rhs: FP)
    {
        let (l_nrows, l_ncols) = self.size();

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                self.a((r, c), self[(r, c)] / rhs);
            }
        }
    }
}

impl<V: MatView> Div<FP> for MatGen<V>
{
    type Output = MatGen<V::OwnColl>;

    fn div(self, rhs: FP) -> MatGen<V::OwnColl>
    {
        let mut mat = self.h_own();
        mat.div_assign(rhs);
        mat
    }
}

impl<V: MatView> Div<FP> for &MatGen<V>
{
    type Output = MatGen<V::OwnColl>;

    fn div(self, rhs: FP) -> MatGen<V::OwnColl>
    {
        let mut mat = self.clone_sz();
        mat.div_assign(rhs);
        mat
    }
}

//

/// Xorshift random number generator
pub struct XOR64(u64);

impl XOR64
{
    pub fn init() -> XOR64
    {
        XOR64(88172645463325252)
    }

    pub fn next(&mut self) -> FP
    {
        const MAX: FP = (1_u128 << 64) as FP;
        self.0 = self.0 ^ (self.0 << 7);
        self.0 = self.0 ^ (self.0 >> 9);

        // [0.0, 1.0)
        (self.0 as FP) / MAX
    }
}

//

#[test]
fn test_set()
{
    {
        let a = Mat::new(3, 3).set_eye(1.);
        let b = Mat::new_like(&a).set_iter(&[
            1., 0., 0.,
            0., 1., 0.,
            0., 0., 1.
        ]);
        println!("{:?}", b.view);
        assert_eq!(a, b);
    }
    {
        let a = Mat::new(2, 4).set_by(|r, c| (r * 4 + c) as FP);
        let b = Mat::new_like(&a).set_iter(&[
            0., 1., 2., 3.,
            4., 5., 6., 7.
        ]);
        println!("{:?}", b.view);
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
        let mut r = XOR64::init();
        let mut a = Mat::new(4, 4);
        let b = Mat::new_like(&a).set_by(|_, _| r.next());
        println!("{}", b);
        a.assign(&b);
        assert_eq!(a, b);
    }
}

#[test]
fn test_slice()
{
    {
        let a = Mat::new(4, 4).set_eye(1.);
        let a = a.slice(1 ..= 2, 1 ..= 2);
        let b = Mat::new(2, 2).set_eye(1.);
        assert_eq!(a, b);
    }
    {
        let mut a = Mat::new(4, 4).set_eye(1.);
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
        let mut a = Mat::new(4, 4).set_eye(1.);
        let b = Mat::new(4, 4).set_iter(&[
            0., 0., 0., 0.,
            0., 1., 0., 0.,
            0., 0., 1., 0.,
            1., 0., 0., 1.
        ]);
        let a1 = a.col(3).clone_sz();
        a.col_mut(0).assign(&a1);
        assert_eq!(a, b);
    }
}

#[test]
fn test_ops()
{
    {
        let a = Mat::new(2, 2).set_eye(1.);
        let b = Mat::new(2, 2).set_iter(&[
            -1., 0.,
            0., -1.
        ]);
        let c = -a;
        assert_eq!(c, b);
    }
    {
        let a = Mat::new(2, 2).set_eye(1.);
        let b = Mat::new(2, 2).set_iter(&[
            -1., 0.,
            0., -1.
        ]);
        let c = -&a;
        assert_eq!(c, b);
        println!("{}", a);
    }
    {
        let a1 = Mat::new(2, 2).set_eye(1.);
        let a2 = Mat::new(2, 2).set_all(1.);
        let b = Mat::new(2, 2).set_iter(&[
            2., 1.,
            1., 2.
        ]);
        let c = a1 + a2;
        assert_eq!(c, b);
    }
    {
        let a1 = Mat::new(2, 2).set_eye(1.);
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
