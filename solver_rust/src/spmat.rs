use super::mat::{MatGen, FP, FP_MINPOS, MatView};

use std::collections::BTreeMap;

/// Matrix
pub type SpMat = MatGen<BTreeMap<usize, FP>>;
/// Matrix slice
pub type SpMatSlice<'a> = MatGen<&'a BTreeMap<usize, FP>>;
/// Matrix slice mutable
pub type SpMatSliMu<'a> = MatGen<&'a BTreeMap<usize, FP>>;

impl MatView for BTreeMap<usize, FP>
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
    fn get_iter_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item=(usize, &mut FP)> + 'a> {
        Box::new(self.iter_mut().map(|(ref_idx, val)| (*ref_idx, val)))
    }
}

impl MatView for &BTreeMap<usize, FP>
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
    fn get_iter_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item=(usize, &mut FP)> + 'a> {
        panic!("cannot borrow immutable as mutable");
    }
}

impl MatView for &mut BTreeMap<usize, FP>
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
    fn get_iter_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item=(usize, &mut FP)> + 'a> {
        Box::new(self.iter_mut().map(|(ref_idx, val)| (*ref_idx, val)))
    }
}

impl Clone for SpMat
{
    fn clone(&self) -> SpMat
    {
        self.clone_sz()
    }
}

#[test]
fn test_spmat()
{
    {
        let a = SpMat::new(2, 2).set_eye(1.);
        let b = SpMat::new(2, 2).set_iter(&[
            -1., 0.,
            0., -1.
        ]);
        let c = -a;
        assert_eq!(c, b);
    }
    {
        let a = SpMat::new(2, 2).set_eye(1.);
        let b = SpMat::new(2, 2).set_iter(&[
            -1., 0.,
            0., -1.
        ]);
        let c = -&a;
        assert_eq!(c, b);
        println!("{:?}", a);
    }
    {
        let a1 = SpMat::new(2, 2).set_eye(1.);
        let a2 = SpMat::new(2, 2).set_all(1.);
        let b = SpMat::new(2, 2).set_iter(&[
            2., 1.,
            1., 2.
        ]);
        let c = a1 + a2;
        assert_eq!(c, b);
    }
    {
        let a1 = SpMat::new(2, 2).set_eye(1.);
        let a2 = SpMat::new(2, 2).set_all(1.);
        let b = SpMat::new(2, 2).set_iter(&[
            2., 1.,
            1., 2.
        ]);
        let c = &a1 + &a2;
        assert_eq!(c, b);
        println!("{}", a1);
    }
}
