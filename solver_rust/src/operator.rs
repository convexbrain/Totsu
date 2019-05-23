
use super::mat::Mat;

// TODO: comments
// TODO: refactoring
pub trait LinOp {
    fn mat(&self) -> Mat;
    fn apply(&self, vec: &Mat) -> Mat {
        assert_eq!(vec.size().1, 1);
        self.mat() * vec // default inefficient implementation
    }
    fn t_apply(&self, vec: &Mat) -> Mat {
        assert_eq!(vec.size().1, 1);
        self.mat().t() * vec // default inefficient implementation
    }
}
