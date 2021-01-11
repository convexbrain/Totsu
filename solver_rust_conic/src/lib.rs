#[cfg(test)]
mod tests;

pub mod solver;     // core, Float

pub mod linalgex;   // core, Float
pub mod f64lapack;  // core, f64(cblas/lapacke)

pub mod cone;       // core, Float
pub mod matop;      // core, Float
pub mod matbuild;   // std,  Float

pub mod nulllogger; // core
pub mod stdlogger;  // std

pub mod probqp;     // std, Float
pub mod probsocp;   // std, Float

// TODO: predef LP/QP/QCQP/SOCP/SDP
// TODO: pub use, mod layout
// TODO: doc
// TODO: thorough tests
