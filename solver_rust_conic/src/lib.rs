#[cfg(test)]
mod tests;

pub mod solver;

pub mod linalgex;
pub mod f64lapack;

pub mod matop;
pub mod cone;

pub mod matbuild;
pub mod logger;

// TODO: predef LP/QP/QCQP/SOCP/SDP
pub mod probqp;
//pub mod probsocp;

// TODO: pub use, mod layout
/*
  solver     core, Float

  linalgex   core, Float
  f64lapack  core, f64,   lapack

  cone       core, Float
  matop      core, Float
  matbuild   std,  Float
  logger     std

  problp     std,  Float
  probqp     std,  Float
  probqcqp   std,  Float
  probsocp   std,  Float
  probsdp    std,  Float
*/ 

// TODO: doc
// TODO: thorough tests
