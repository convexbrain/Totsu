//! Loggers of solver progress

mod nulllogger; // core

#[cfg(feature = "std")]
mod stdlogger;  // std

pub use nulllogger::*;

#[cfg(feature = "std")]
pub use stdlogger::*;
