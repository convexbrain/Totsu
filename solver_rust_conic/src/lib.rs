#[cfg(test)]
mod tests;

mod solver;
mod matop;
mod cone;
mod linalg;

pub use crate::solver::{Solver, SolverParam, Operator, Cone};

// TODO: LP/QP/QCQP/SOCP/SDP
pub use crate::matop::{MatOp};
pub use crate::cone::{ConePSD};
