//! loggers of solver progress

mod nulllogger; // core
mod stdlogger;  // std

pub use nulllogger::*;
pub use stdlogger::*;
