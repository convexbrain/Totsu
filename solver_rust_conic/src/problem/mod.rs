//! Pre-defined optimization problems

mod lp;   // std, Float
mod qp;   // std, Float
mod qcqp; // std, Float
mod socp; // std, Float
mod sdp;  // std, Float

pub use lp::ProbLP;
pub use qp::ProbQP;
pub use qcqp::ProbQCQP;
pub use socp::ProbSOCP;
pub use sdp::ProbSDP;
