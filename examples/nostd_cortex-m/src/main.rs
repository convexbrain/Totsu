// Runnable on QEMU ARM

#![no_main]
#![no_std]

use cortex_m_rt::entry;
use cortex_m_semihosting::debug;
use cortex_m_semihosting::{hprint, hprintln};
use panic_semihosting as _;

#[entry]
fn main() -> !
{
    hprintln!("run").unwrap();

    test_lp();
    
    hprintln!("exit").unwrap();
    debug::exit(debug::EXIT_SUCCESS);
    loop {}
}

//

use totsu::prelude::*;
use num_traits::Float;

//

/// Logger using `hprint!` macro
pub struct HPrintLogger;

impl core::fmt::Write for HPrintLogger
{
    fn write_str(&mut self, s: &str) -> Result<(), core::fmt::Error>
    {
        if let Ok(_) = hprint!("{}", s) {
            Ok(())
        }
        else {
            Err(core::fmt::Error)
        }
    }
}

//

fn test_lp()
{
    type LA = FloatGeneric<f64>;
    type AMatOp<'a> = MatOp<'a, LA, f64>;
    type AConeRPos = ConeRPos<f64>;
    type ASolver = Solver<LA, f64>;

    let n = 2; // x, y
    let m = 3;

    let op_c = AMatOp::new(MatType::General(n, 1), &[ // NOTE: Column-major
        -1., 0.,
    ]);

    let op_a = AMatOp::new(MatType::General(m, n), &[ // NOTE: Column-major
         4., -1., -1., // Column 0
        -1.,  4., -1., // Column 1
    ]);

    let op_b = AMatOp::new(MatType::General(m, 1), &[
        6., 6., 1.,
    ]);

    let cone = AConeRPos::new();

    // objective: going to +x direction as much as possible
    // constraints: inside of triangle whose vertices are (2, 2), (-2, 1) and (1, -2)
    // exact solution: (2, 2)

    let s = ASolver::new().par(|p| {
        p.max_iter = Some(100_000);
        p.log_period = Some(10);
    });

    hprintln!("query_worklen -> {}", ASolver::query_worklen(op_a.size())).unwrap();
    let mut solver_w = [0.; 48];

    let rslt = s.solve((op_c, op_a, op_b, cone, &mut solver_w), HPrintLogger).unwrap();
    hprintln!("solve -> {:?}", rslt.0).unwrap();

    assert!((rslt.0[0] - 2.).abs() <= 1e-3);
    assert!((rslt.0[1] - 2.).abs() <= 1e-3);
}
