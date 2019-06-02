/*!
Matrix

[`Mat`](type.Mat.html) is primal type used to make and own matrix value.
See also [`MatGen`](struct.MatGen.html) for supported methods.
*/

pub use super::matgen::{FP, FP_EPSILON, FP_MINPOS};
use super::matgen::{MatGen, MatView};

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
        println!("{:?}", b);
        assert_eq!(a, b);
    }
    {
        let a = Mat::new(2, 4).set_by(|r, c| (r * 4 + c) as FP);
        let b = Mat::new_like(&a).set_iter(&[
            0., 1., 2., 3.,
            4., 5., 6., 7.
        ]);
        println!("{:?}", b);
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
