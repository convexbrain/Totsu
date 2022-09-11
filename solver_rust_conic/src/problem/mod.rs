//! Pre-defined optimization problems
//! 
//! This module relies on dynamic heap allocation.

mod lp;   // std, Float
mod qp;   // std, Float
/* TODO: restore
mod qcqp; // std, Float
mod socp; // std, Float
mod sdp;  // std, Float
 */

pub use lp::ProbLP;
pub use qp::ProbQP;
/* TODO: restore
pub use qcqp::ProbQCQP;
pub use socp::ProbSOCP;
pub use sdp::ProbSDP;
 */
