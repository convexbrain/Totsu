#[cfg(test)]
mod tests;

pub mod solver;
pub mod matop;

pub mod prelude {
    pub use crate::solver::{SolverParam, Solver};
}

pub mod predef {
    pub use crate::matop::MatOp;
}
