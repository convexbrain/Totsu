//! Loggers of solver progress

mod nulllogger; // core

#[cfg(not(feature = "nostd"))]
mod stdlogger;  // std

pub use nulllogger::*;

#[cfg(not(feature = "nostd"))]
pub use stdlogger::*;
