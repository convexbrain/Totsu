#[cfg(test)]
mod tests;

pub mod solver;
pub mod matop;
pub mod proj;
pub mod linalg;

pub mod prelude {
    pub use crate::solver::{SolverParam, Solver, Operator, Projection};
}

pub mod predef {
    pub use crate::matop::{MatOp};
    pub use crate::proj::{ProjPSD};
}
