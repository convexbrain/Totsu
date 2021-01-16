use num::Float;
use crate::solver::SolverError;

pub trait Cone<F: Float>
{
    // x.len() shall be op_a.size().0
    fn proj(&mut self, eps_zero: F, x: &mut[F]) -> Result<(), SolverError>;
    fn dual_proj(&mut self, eps_zero: F, x: &mut[F]) -> Result<(), SolverError>
    {
        self.proj(eps_zero, x) // Self-dual cone
    }
}

//

mod zero; // core, Float
mod rpos; // core, Float
mod soc;  // core, Float
mod psd;  // core, Float

pub use zero::*;
pub use rpos::*;
pub use soc::*;
pub use psd::*;
