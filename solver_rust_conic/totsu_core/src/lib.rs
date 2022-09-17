#![no_std]

pub mod solver;

//

mod linalg_ex;

pub use linalg_ex::*;

//

mod floatgeneric;

pub use floatgeneric::*;

//

mod matop;

pub use matop::*;

//

mod cone_zero;
mod cone_rpos;
mod cone_soc;
mod cone_rotsoc;
mod cone_psd;

pub use cone_zero::*;
pub use cone_rpos::*;
pub use cone_soc::*;
pub use cone_rotsoc::*;
pub use cone_psd::*;
