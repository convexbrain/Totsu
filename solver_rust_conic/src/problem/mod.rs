//! Pre-defined optimization problems
//! 
//! This module relies on dynamic heap allocation.

mod lp;   // std, Float
mod qp;   // std, Float
mod qcqp; // std, Float
mod socp; // std, Float
/* TODO: restore
mod sdp;  // std, Float
 */

pub use lp::ProbLP;
pub use qp::ProbQP;
pub use qcqp::ProbQCQP;
pub use socp::ProbSOCP;
/* TODO: restore
pub use sdp::ProbSDP;
 */
