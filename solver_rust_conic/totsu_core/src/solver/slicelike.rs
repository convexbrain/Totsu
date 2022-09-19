use num_traits::Float;
use core::ops::{Deref, DerefMut, Drop};

//

/// Slice-like trait for a vector of linear algebra.
/// 
/// [`crate::solver::LinAlg::Sl`] shall have this trait boundary.
pub trait SliceLike
{
    /// Floating point data type of slice elements.
    /// 
    /// This will be the same as [`crate::solver::LinAlg::F`].
    type F: Float;

    /// Borrows a slice from which [`SliceLike`] is created, and a reference of the [`SliceLike`] is wrapped by a [`SliceRef`].
    /// 
    /// [`SliceLike::drop`] shall be called when the [`SliceRef`] drops,
    /// which means the borrowal ends and the [`SliceLike`] can be dropped safely.
    /// 
    /// Returns the [`SliceRef`] wrapping the reference of [`SliceLike`].
    /// * `s` is a reference of the original slice.
    fn new_ref(s: &[Self::F]) -> SliceRef<'_, Self>;

    /// Mutable version of [`SliceLike::new_ref`].
    fn new_mut(s: &mut[Self::F]) -> SliceMut<'_, Self>;

    /// Divides one reference of a [`SliceLike`] into two at an index `mid`.
    /// 
    /// Returns all indices from [`0`, `mid`) as the first and all indices from [`mid`, [`SliceLike::len()`]) as the second.
    /// * `mid` is an index for splitting.
    fn split_ref(&self, mid: usize) -> (SliceRef<'_, Self>, SliceRef<'_, Self>);

    /// Mutable version of [`SliceLike::split_ref`].
    fn split_mut(&mut self, mid: usize) -> (SliceMut<'_, Self>, SliceMut<'_, Self>);

    /// Drops a [`SliceLike`].
    /// 
    /// This shall be called when [`SliceRef`] or [`SliceMut`] drops.
    /// At this point the referenced [`SliceLike`] can be dropped safely.
    fn drop(&self);

    fn len(&self) -> usize;

    fn get_ref(&self) -> &[Self::F];
    fn get_mut(&mut self) -> &mut[Self::F];

    fn get(&self, idx: usize) -> Self::F
    {
        let (_, spl) = self.split_ref(idx);
        let (ind, _) = spl.split_ref(1);
        ind.get_ref()[0]
    }
    fn set(&mut self, idx: usize, val: Self::F)
    {
        let (_, mut spl) = self.split_mut(idx);
        let (mut ind, _) = spl.split_mut(1);
        ind.get_mut()[0] = val;
    }
}

/// Wrapping a reference of [`SliceLike`].
#[derive(Debug)] // NOTE: Do not derive clone, or the functionality of SliceLike::drop may break.
pub struct SliceRef<'a, S: SliceLike + ?Sized>
{
    s: &'a S,
}

impl<'a, S: SliceLike + ?Sized> SliceRef<'a, S>
{
    /// Wraps a reference of [`SliceLike`] and creates a [`SliceRef`].
    /// This is **unsafe** and intended to be used only by [`SliceLike::new_ref`] and [`SliceLike::split_ref`] implementor.
    /// 
    /// Returns the [`SliceRef`].
    /// * `s` is a reference of [`SliceLike`].
    pub unsafe fn new(s: &'a S) -> Self
    {
        SliceRef {s}
    }
}

impl<'a, S: SliceLike + ?Sized> Deref for SliceRef<'a, S>
{
    type Target = S;
    fn deref(&self) -> &Self::Target {self.s}
}

impl<'a, S: SliceLike + ?Sized> Drop for SliceRef<'a, S>
{
    fn drop(&mut self) {
        self.s.drop();
    }
}

/// Wrapping a mutable reference of [`SliceLike`].
#[derive(Debug)]
pub struct SliceMut<'a, S: SliceLike + ?Sized>
{
    s: &'a mut S,
}

impl<'a, S: SliceLike + ?Sized> SliceMut<'a, S>
{
    /// Wraps a mutable reference of [`SliceLike`] and creates a [`SliceMut`].
    /// This is **unsafe** and intended to be used only by [`SliceLike::new_mut`] and [`SliceLike::split_mut`] implementor.
    /// 
    /// Returns the [`SliceMut`].
    /// * `s` is a mutable reference of [`SliceLike`].
    pub unsafe fn new(s: &'a mut S) -> Self
    {
        SliceMut {s}
    }
}

impl<'a, S: SliceLike + ?Sized> Deref for SliceMut<'a, S>
{
    type Target = S;
    fn deref(&self) -> &Self::Target {self.s}
}

impl<'a, S: SliceLike + ?Sized> DerefMut for SliceMut<'a, S>
{
    fn deref_mut(&mut self) -> &mut Self::Target {self.s}
}

impl<'a, S: SliceLike + ?Sized> Drop for SliceMut<'a, S>
{
    fn drop(&mut self) {
        self.s.drop();
    }
}

/// Splits a [`SliceRef`] into multiple ones.
/// 
/// ```
/// use totsu_core::solver::{SliceLike, SliceRef, LinAlg};
/// use totsu_core::{splitm, FloatGeneric};
/// 
/// fn func<L: LinAlg>(x: SliceRef<L::Sl>) {
///     splitm!(x, (x1; 4), (x2; 5), (x3; 6));
///     assert_eq!(x1.len(), 4);
///     assert_eq!(x2.len(), 5);
///     assert_eq!(x3.len(), 6);
/// }
/// 
/// type L = FloatGeneric<f64>;
/// let a = &[0.; 20];
/// let x = <L as LinAlg>::Sl::new_ref(a);
/// func::<L>(x);
/// ```
#[macro_export]
macro_rules! splitm {
    ($slice:expr, $( ($var:ident; $len:expr) ),+ ) => {
        let (_, _splitm_rest) = $slice.split_ref(0);
        $(
            let ($var, _splitm_rest) = _splitm_rest.split_ref($len);
        )*
        drop(_splitm_rest);
    };
}

/// Splits a [`SliceMut`] into multiple ones.
/// 
/// ```
/// use totsu_core::solver::{SliceLike, SliceMut, LinAlg};
/// use totsu_core::{splitm_mut, FloatGeneric};
/// 
/// fn func<L: LinAlg>(mut x: SliceMut<L::Sl>) {
///     splitm_mut!(x, (x1; 4), (x2; 5), (x3; 6));
///     assert_eq!(x1.len(), 4);
///     assert_eq!(x2.len(), 5);
///     assert_eq!(x3.len(), 6);
/// }
/// 
/// type L = FloatGeneric<f64>;
/// let a = &mut[0.; 20];
/// let x = <L as LinAlg>::Sl::new_mut(a);
/// func::<L>(x);
/// ```
#[macro_export]
macro_rules! splitm_mut {
    ($slice:expr, $( ($var:ident; $len:expr) ),+ ) => {
        let (_, mut _splitm_rest) = $slice.split_mut(0);
        $(
            let (mut $var, mut _splitm_rest) = _splitm_rest.split_mut($len);
        )*
        drop(_splitm_rest);
    };
}

//

impl<F: Float> SliceLike for [F]
{
    type F = F;

    fn new_ref(s: &[F]) -> SliceRef<'_, Self>
    {
        SliceRef {s}
    }

    fn new_mut(s: &mut[F]) -> SliceMut<'_, Self>
    {
        SliceMut {s}
    }

    fn split_ref(&self, mid: usize) -> (SliceRef<'_, Self>, SliceRef<'_, Self>)
    {
        let s = self.split_at(mid);
        (SliceRef {s: s.0}, SliceRef {s: s.1})
    }

    fn split_mut(&mut self, mid: usize) -> (SliceMut<'_, Self>, SliceMut<'_, Self>)
    {
        let s = self.split_at_mut(mid);
        (SliceMut {s: s.0}, SliceMut {s: s.1})
    }

    fn drop(&self)
    {
    }

    fn len(&self) -> usize
    {
        self.len()
    }

    fn get_ref(&self) -> &[F]
    {
        self
    }

    fn get_mut(&mut self) -> &mut[F]
    {
        self
    }
}
