
use super::mat::Mat;

// TODO: comments
pub trait LinOp {
    fn mat(&self) -> Mat;
    fn apply(&self, vec: &Mat) -> Mat {
        assert_eq!(vec.size().1, 1);
        self.mat() * vec
    }
    fn t_apply(&self, vec: &Mat) -> Mat {
        assert_eq!(vec.size().1, 1);
        self.mat().t() * vec
    }
}
