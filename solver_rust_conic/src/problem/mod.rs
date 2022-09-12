//! Pre-defined optimization problems
//! 
//! This module relies on dynamic heap allocation.

mod lp;   // std, Float
mod qp;   // std, Float
mod qcqp; // std, Float

/* TODO: restore
mod socp; // std, Float
mod sdp;  // std, Float
 */

pub use lp::ProbLP;
pub use qp::ProbQP;
pub use qcqp::ProbQCQP;
/* TODO: restore
pub use socp::ProbSOCP;
pub use sdp::ProbSDP;
 */
