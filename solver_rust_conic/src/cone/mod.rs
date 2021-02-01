use num::Float;

pub trait Cone<F: Float>
{
    // x.len() shall be op_a.size().0
    fn proj(&mut self, dual_cone: bool, eps_zero: F, x: &mut[F]) -> Result<(), ()>;
    fn product_group<G: Fn(&mut[F]) + Copy>(&self, dp_tau: &mut[F], group: G);
}

//

mod zero;    // core, Float
mod rpos;    // core, Float
mod soc;     // core, Float
mod rotsoc;  // core, Float
mod psd;     // core, Float

pub use zero::*;
pub use rpos::*;
pub use soc::*;
pub use rotsoc::*;
pub use psd::*;
