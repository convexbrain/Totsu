#[cfg(test)]
mod tests;

pub mod solver;

pub mod linalgex;
pub mod f64_lapack;

pub mod matop;
pub mod cone;

pub mod matbuild;
pub mod logger;

pub mod probqp;

// TODO: pub use, mod layout
/*
  solver

  linalgex    f64_lapack

         cone
         matop       matbuild
                                   logger
        
              problp
              probqp
              probqcqp
              probsocp
              probsdp
*/ 


// TODO: LP/QP/QCQP/SOCP/SDP
// TODO: doc
// TODO: thorough tests
