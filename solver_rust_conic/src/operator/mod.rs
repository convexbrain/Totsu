use num::Float;

pub trait Operator<F: Float>
{
    fn size(&self) -> (usize, usize);

    // y = alpha * Op * x + beta * y
    // x.len() shall be size().1
    // y.len() shall be size().0
    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F]);

    // y = alpha * Op^T * x + beta * y
    // x.len() shall be size().0
    // y.len() shall be size().1
    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F]);
}

//

mod matop;    // core, Float
mod matbuild; // std,  Float

pub use matop::*;
pub use matbuild::*;
