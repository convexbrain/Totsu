pub mod solver; // core, Float

pub mod linalg;
pub mod operator;
pub mod cone;
pub mod logger;
pub mod problem;

#[cfg(test)]
extern crate intel_mkl_src;

// TODO: no-std
// TODO: examples
// TODO: doc
