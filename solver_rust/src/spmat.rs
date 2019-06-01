use super::mat::{MatGen, FP, FP_MINPOS, View};

use std::collections::BTreeMap;

/// Matrix
pub type SpMat = MatGen<BTreeMap<usize, FP>>;
/// Matrix slice
pub type SpMatSlice<'a> = MatGen<&'a BTreeMap<usize, FP>>;
/// Matrix slice mutable
pub type SpMatSliMu<'a> = MatGen<&'a BTreeMap<usize, FP>>;

impl View for BTreeMap<usize, FP>
{
    type OwnColl = BTreeMap<usize, FP>;

    fn new_own(_sz: usize) -> Self::OwnColl
    {
        BTreeMap::new()
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
        self.get(&i).unwrap_or(&0.)
    }
    fn get_index_mut(&mut self, i: usize) -> &mut FP
    {
        self.entry(i).or_insert(0.)
    }
    fn put(&mut self, i: usize, val: FP)
    {
        if val.abs() < FP_MINPOS {
            self.remove(&i);
        }
        else {
            *self.get_index_mut(i) = val;
        }
    }
}

impl View for &BTreeMap<usize, FP>
{
    type OwnColl = BTreeMap<usize, FP>;

    fn new_own(_sz: usize) -> Self::OwnColl
    {
        BTreeMap::new()
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
        self.get(&i).unwrap_or(&0.)
    }
    fn get_index_mut(&mut self, _i: usize) -> &mut FP
    {
        panic!("cannot borrow immutable as mutable");
    }
    fn put(&mut self, _i: usize, _val: FP)
    {
        panic!("cannot borrow immutable as mutable");
    }
}

impl View for &mut BTreeMap<usize, FP>
{
    type OwnColl = BTreeMap<usize, FP>;

    fn new_own(_sz: usize) -> Self::OwnColl
    {
        BTreeMap::new()
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
        self.get(&i).unwrap_or(&0.)
    }
    fn get_index_mut(&mut self, i: usize) -> &mut FP
    {
        self.entry(i).or_insert(0.)
    }
    fn put(&mut self, i: usize, val: FP)
    {
        if val.abs() < FP_MINPOS {
            self.remove(&i);
        }
        else {
            *self.get_index_mut(i) = val;
        }
    }
}

impl Clone for SpMat
{
    fn clone(&self) -> SpMat
    {
        self.clone_sz()
    }
}
