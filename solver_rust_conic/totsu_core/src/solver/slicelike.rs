use num_traits::Float;
use core::ops::{Deref, DerefMut, Drop};

//

// TODO: doc
pub trait SliceLike
{
    type F: Float;

    fn new_ref(s: &[Self::F]) -> SliceRef<'_, Self>;
    fn new_mut(s: &mut[Self::F]) -> SliceMut<'_, Self>;

    fn split_ref(&self, mid: usize) -> (SliceRef<'_, Self>, SliceRef<'_, Self>);
    fn split_mut(&mut self, mid: usize) -> (SliceMut<'_, Self>, SliceMut<'_, Self>);

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

// TODO: doc
#[derive(Debug)] // NOTE: Do not derive clone, or the functionality of SliceLike::drop may break.
pub struct SliceRef<'a, S: SliceLike + ?Sized>
{
    s: &'a S,
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

// TODO: doc
#[derive(Debug)]
pub struct SliceMut<'a, S: SliceLike + ?Sized>
{
    s: &'a mut S,
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

// TODO: https://stackoverflow.com/questions/53204327/how-to-have-a-private-part-of-a-trait
impl<'a, S: SliceLike + ?Sized> SliceRef<'a, S>
{
    pub fn new(s: &'a S) -> Self
    {
        SliceRef {s}
    }
}
impl<'a, S: SliceLike + ?Sized> SliceMut<'a, S>
{
    pub fn new(s: &'a mut S) -> Self
    {
        SliceMut {s}
    }
}

// TODO: public?
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

// TODO: public?
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
