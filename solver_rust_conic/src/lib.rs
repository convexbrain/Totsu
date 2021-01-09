#[cfg(test)]
mod tests;

pub mod solver;

pub mod matop;
pub mod cone;
pub mod linalg;

pub mod matbuild;
pub mod logger;

pub mod probqp;

// TODO: pub use, mod layout
/*
* totsu
  * solver
  * float_generic
    * linalg
    * cone
    * matop
    * matbuild
  * f64_lapack
    * linalg
    * use float_generic::cone
    * use float_generic::matop
    * use float_generic::matbuild
  * logger
  * predef
    * problp
    * probqp
    * probqcqp
    * probsocp
    * probsdp
*/ 


// TODO: LP/QP/QCQP/SOCP/SDP
// TODO: doc
// TODO: thorough tests
