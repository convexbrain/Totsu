#[cfg(test)]
mod tests;

mod solver;
mod matop;
mod proj;
mod linalg;

pub use crate::solver::{Solver, SolverParam, Operator, Projection};

// TODO: LP/QP/QCQP/SOCP/SDP
pub use crate::matop::{MatOp};
pub use crate::proj::{ProjPSD};
