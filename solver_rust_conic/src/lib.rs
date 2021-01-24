pub mod solver; // core, Float
mod utils;

pub mod linalg;
pub mod operator;
pub mod cone;
pub mod logger;
pub mod problem;

pub mod prelude // core, Float
{
    pub use super::solver::{Solver, SolverError, SolverParam};
    pub use super::linalg::{LinAlg, LinAlgEx, FloatGeneric};
    pub use super::operator::{Operator, MatType, MatOp};
    pub use super::cone::{Cone, ConeZero, ConeRPos, ConeSOC, ConePSD};
    pub use super::logger::NullLogger;
}

// TODO: no-std
// MUST TODO: examples
// MUST TODO: doc, readme
// TODO: more tests
