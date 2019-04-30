/*!
Matrix

[`Mat`](type.Mat.html) is primal type used to make and own matrix value.
See also [`MatGen`](struct.MatGen.html) for supported methods.
*/

use std::cmp::PartialEq;
use std::ops::{Range, RangeBounds, Bound};
use std::ops::{Neg, Add, Mul, Sub, Div, AddAssign, SubAssign, MulAssign, DivAssign};
use std::fmt;

/// Scalar floating point type
pub type FP = f64;
pub use std::f64::EPSILON as FP_EPSILON;
pub use std::f64::MIN_POSITIVE as FP_MINPOS;

/// Matrix elements collection
type Coll = Vec<FP>;
//type Coll = std::collections::HashMap<usize, FP>;

/// Matrix
pub type Mat = MatGen<Coll>;
/// Matrix slice
pub type MatSlice<'a> = MatGen<&'a Coll>;
/// Matrix slice mutable
pub type MatSliMu<'a> = MatGen<&'a mut Coll>;

/// Ownership view of matrix elements collection
pub trait View {
    fn get_ref(&self) -> &Coll;
    fn get_mut(&mut self) -> &mut Coll;
    fn get_len(&self) -> usize;
    fn get_own(self) -> Coll;
    fn is_own(&self) -> bool;
}

impl View for Coll
{
    fn get_ref(&self) -> &Coll
    {
        &self
    }
    fn get_mut(&mut self) -> &mut Coll
    {
        self
    }
    fn get_len(&self) -> usize
    {
        self.len()
    }
    fn get_own(self) -> Coll
    {
        self
    }
    fn is_own(&self) -> bool
    {
        true
    }
}

impl View for &Coll
{
    fn get_ref(&self) -> &Coll
    {
        self
    }
    fn get_mut(&mut self) -> &mut Coll
    {
        panic!("cannot borrow immutable as mutable");
    }
    fn get_len(&self) -> usize
    {
        self.len()
    }
    fn get_own(self) -> Coll
    {
        panic!("cannot own immutable");
    }
    fn is_own(&self) -> bool
    {
        false
    }
}

impl View for &mut Coll
{
    fn get_ref(&self) -> &Coll
    {
        self
    }
    fn get_mut(&mut self) -> &mut Coll
    {
        self
    }
    fn get_len(&self) -> usize
    {
        self.len()
    }
    fn get_own(self) -> Coll
    {
        panic!("cannot own mutable");
    }
    fn is_own(&self) -> bool
    {
        false
    }
}

/// Generic struct of matrix
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
    fn h_index(&self, row: usize, col: usize) -> usize
    {
        if !self.transposed {
            self.offset + self.stride * col + row
        }
        else {
            self.offset + self.stride * row + col
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
            self.clone_sz()
        }
    }
    //
    /// *new* - Makes a matrix.
    pub fn new(nrows: usize, ncols: usize) -> Mat
    {
        MatGen {
            nrows,
            ncols,
            offset: 0,
            stride: nrows,
            transposed: false,
            view: vec![0.; nrows * ncols]
            //view: Coll::new()
        }
    }
    /// *new* - Makes a matrix of the same size.
    pub fn new_like<V2: View>(mat: &MatGen<V2>) -> Mat
    {
        let (nrows, ncols) = mat.size();

        Mat::new(nrows, ncols)
    }
    /// *new* - Makes a column vector.
    pub fn new_vec(nrows: usize) -> Mat
    {
        Mat::new(nrows, 1)
    }
    //
    /// Gets a value of an element
    pub fn get(&self, row: usize, col: usize) -> FP
    {
        let i = self.h_index(row, col);

        self.view.get_ref()[i]
        //*self.view.get_ref().get(&i).unwrap_or(&0.)
    }
    /// Puts a value into an element
    pub fn put(&mut self, row: usize, col: usize, val: FP)
    {
        let i = self.h_index(row, col);

        self.view.get_mut()[i] = val;
        /*
        if val.abs() < FP_MINPOS {
            self.view.get_mut().remove(&i);
        }
        else {
            self.view.get_mut().insert(i, val);
        }
        */
    }
    //
    /// *slice* - Slice block reference.
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
    /// *slice* - Slice block mutable reference.
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
    /// *slice* - Row vectors reference.
    pub fn rows<RR>(&self, rows: RR) -> MatSlice
    where RR: RangeBounds<usize>
    {
        self.slice(rows, ..)
    }
    /// *slice* - Column vectors reference.
    pub fn cols<CR>(&self, cols: CR) -> MatSlice
    where CR: RangeBounds<usize>
    {
        self.slice(.., cols)
    }
    /// *slice* - A row vector reference.
    pub fn row(&self, r: usize) -> MatSlice
    {
        self.rows(r ..= r)
    }
    /// *slice* - A column vector reference.
    pub fn col(&self, c: usize) -> MatSlice
    {
        self.cols(c ..= c)
    }
    /// *slice* - Row vectors mutable reference.
    pub fn rows_mut<RR>(&mut self, rows: RR) -> MatSliMu
    where RR: RangeBounds<usize>
    {
        self.slice_mut(rows, ..)
    }
    /// *slice* - Column vectors mutable reference.
    pub fn cols_mut<CR>(&mut self, cols: CR) -> MatSliMu
    where CR: RangeBounds<usize>
    {
        self.slice_mut(.., cols)
    }
    /// *slice* - A row vector mutable reference.
    pub fn row_mut(&mut self, r: usize) -> MatSliMu
    {
        self.rows_mut(r ..= r)
    }
    /// *slice* - A column vector mutable reference.
    pub fn col_mut(&mut self, c: usize) -> MatSliMu
    {
        self.cols_mut(c ..= c)
    }
    /// *slice* - Whole reference.
    pub fn as_slice(&self) -> MatSlice
    {
        self.slice(.., ..)
    }
    /// *slice* - Whole mutable reference.
    pub fn as_slice_mut(&mut self) -> MatSliMu
    {
        self.slice_mut(.., ..)
    }
    /// *slice* - Transopsed reference.
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
    /// *slice* - Transopsed mutable reference.
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
    pub fn clone_sz(&self) -> Mat
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
                view: self.view.get_ref().clone()
            }
        }
        else {
            let mut mat = Mat::new(l_nrows, l_ncols);
            mat.assign(self);
            mat
        }
    }
    /// *clone* - Clone into diagonal matrix.
    pub fn clone_diag(&self) -> Mat
    {
        let (l_nrows, l_ncols) = self.size();
        assert_eq!(l_ncols, 1);

        let mut mat = Mat::new(l_nrows, l_nrows);

        for r in 0 .. l_nrows {
            mat.put(r, r, self.get(r, 0));
        }

        mat
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
                    self.put(r, c, value);
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
                self.put(r, c, *i.next().unwrap_or(&0.));
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
    /// *assign* - Assign matrix.
    pub fn assign<V2: View>(&mut self, rhs: &MatGen<V2>)
    {
        let (l_nrows, l_ncols) = self.size();
        let (r_nrows, r_ncols) = rhs.size();

        assert_eq!(l_nrows, r_nrows);
        assert_eq!(l_ncols, r_ncols);
        
        self.assign_by(|r, c| Some(rhs.get(r, c)));
    }
    //
    /// Returns p=2 norm squared.
    pub fn norm_p2sq(&self) -> FP
    {
        let (l_nrows, l_ncols) = self.size();

        let mut sum = 0.;

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                sum += self.get(r, c) * self.get(r, c);
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
            sum += self.get(i, i);
        }

        sum
    }
    /// Returns inner product.
    pub fn prod<V2: View>(&self, rhs: &MatGen<V2>) -> FP
    {
        let (l_nrows, l_ncols) = self.size();
        let (r_nrows, r_ncols) = rhs.size();

        assert_eq!(l_nrows, r_nrows);
        assert_eq!(l_ncols, r_ncols);

        let mut sum = 0.;

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                sum += self.get(r, c) * rhs.get(r, c);
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

        let mut m = self.get(0, 0);

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                if self.get(r, c) > m {
                    m = self.get(r, c);
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
        
        let mut m = self.get(0, 0);

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                if self.get(r, c) < m {
                    m = self.get(r, c);
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
}

//

impl<V: View> fmt::LowerExp for MatGen<V>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error>
    {
        let (l_nrows, l_ncols) = self.size();

        writeln!(f, "[")?;
        for r in 0 .. l_nrows {
            for c in 0 .. l_ncols {
                write!(f, "  {:.precision$e},", self.get(r, c), precision = f.precision().unwrap_or(3))?;
            }
            writeln!(f)?;
        }
        write!(f, "]")
    }
}

impl<V: View> fmt::Display for MatGen<V>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error>
    {
        writeln!(f, "{:.3e}", self)
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
                if self.get(r, c) != other.get(r, c) {
                    return false;
                }
            }
        }

        true
    }
}

//

impl Clone for Mat
{
    fn clone(&self) -> Mat
    {
        self.clone_sz()
    }
}

//

/// Helper matrix accessor for operator overload
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
        self.get(row, col)
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
                mat.put(r, c, -mat.get(r, c));
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
        self.clone_sz().neg()
    }
}

//

impl<V: View, T: MatAcc> AddAssign<T> for MatGen<V>
{
    fn add_assign(&mut self, rhs: T)
    {
        let (l_nrows, l_ncols) = self.size();

        assert_eq!((l_nrows, l_ncols), rhs.acc_size());

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                self.put(r, c, self.get(r, c) + rhs.acc_get(r, c));
            }
        }
    }
}

impl<V: View> AddAssign<FP> for MatGen<V>
{
    fn add_assign(&mut self, rhs: FP)
    {
        let (l_nrows, l_ncols) = self.size();

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                self.put(r, c, self.get(r, c) + rhs);
            }
        }
    }
}

impl<V: View, T: MatAcc> Add<T> for MatGen<V>
{
    type Output = Mat;

    fn add(self, rhs: T) -> Mat
    {
        let mut mat = self.h_own();
        mat.add_assign(rhs);
        mat
    }
}

impl<V: View, T: MatAcc> Add<T> for &MatGen<V>
{
    type Output = Mat;

    fn add(self, rhs: T) -> Mat
    {
        self.clone_sz().add(rhs)
    }
}

impl<V: View> Add<FP> for MatGen<V>
{
    type Output = Mat;

    fn add(self, rhs: FP) -> Mat
    {
        let mut mat = self.h_own();
        mat.add_assign(rhs);
        mat
    }
}

impl<V: View> Add<FP> for &MatGen<V>
{
    type Output = Mat;

    fn add(self, rhs: FP) -> Mat
    {
        self.clone_sz().add(rhs)
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

impl<V: View, T: MatAcc> SubAssign<T> for MatGen<V>
{
    fn sub_assign(&mut self, rhs: T)
    {
        let (l_nrows, l_ncols) = self.size();

        assert_eq!((l_nrows, l_ncols), rhs.acc_size());

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                self.put(r, c, self.get(r, c) - rhs.acc_get(r, c));
            }
        }
    }
}

impl<V: View> SubAssign<FP> for MatGen<V>
{
    fn sub_assign(&mut self, rhs: FP)
    {
        let (l_nrows, l_ncols) = self.size();

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                self.put(r, c, self.get(r, c) - rhs);
            }
        }
    }
}

impl<V: View, T: MatAcc> Sub<T> for MatGen<V>
{
    type Output = Mat;

    fn sub(self, rhs: T) -> Mat
    {
        let mut mat = self.h_own();
        mat.sub_assign(rhs);
        mat
    }
}

impl<V: View, T: MatAcc> Sub<T> for &MatGen<V>
{
    type Output = Mat;

    fn sub(self, rhs: T) -> Mat
    {
        self.clone_sz().sub(rhs)
    }
}

impl<V: View> Sub<FP> for MatGen<V>
{
    type Output = Mat;

    fn sub(self, rhs: FP) -> Mat
    {
        let mut mat = self.h_own();
        mat.sub_assign(rhs);
        mat
    }
}

impl<V: View> Sub<FP> for &MatGen<V>
{
    type Output = Mat;

    fn sub(self, rhs: FP) -> Mat
    {
        self.clone_sz().sub(rhs)
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

impl<V: View> MulAssign<FP> for MatGen<V>
{
    fn mul_assign(&mut self, rhs: FP)
    {
        let (l_nrows, l_ncols) = self.size();

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                self.put(r, c, self.get(r, c) * rhs);
            }
        }
    }
}

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
                    v += self.get(r, k) * rhs.acc_get(k, c);
                }
                mat.put(r, c, v);
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
        mat.mul_assign(rhs);
        mat
    }
}

impl<V: View> Mul<FP> for &MatGen<V>
{
    type Output = Mat;

    fn mul(self, rhs: FP) -> Mat
    {
        self.clone_sz().mul(rhs)
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

impl<V: View> DivAssign<FP> for MatGen<V>
{
    fn div_assign(&mut self, rhs: FP)
    {
        let (l_nrows, l_ncols) = self.size();

        for c in 0 .. l_ncols {
            for r in 0 .. l_nrows {
                self.put(r, c, self.get(r, c) / rhs);
            }
        }
    }
}

impl<V: View> Div<FP> for MatGen<V>
{
    type Output = Mat;

    fn div(self, rhs: FP) -> Mat
    {
        let mut mat = self.h_own();
        mat.div_assign(rhs);
        mat
    }
}

impl<V: View> Div<FP> for &MatGen<V>
{
    type Output = Mat;

    fn div(self, rhs: FP) -> Mat
    {
        self.clone_sz().div(rhs)
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
