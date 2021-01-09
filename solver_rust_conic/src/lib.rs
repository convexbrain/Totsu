#[cfg(test)]
mod tests;

pub mod solver;

pub mod matop;
pub mod cone;
pub mod linalg;

pub mod matbuild;
pub mod logger;

//pub mod qp;

// TODO: pub use, mod layout
/*
totsu --- solver
       |
       -- float_generic --- linalg
       |                 |
       |                 -- matop
       |                 |
       |                 -- cone
       |
       -- f64_lapack --- linalg
       |              |
       |              -- use f64_generic::matop
       |              |
       |              -- use f64_generic::cone
       |
       -- logger
*/ 


// TODO: LP/QP/QCQP/SOCP/SDP
// TODO: doc
