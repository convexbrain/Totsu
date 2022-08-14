// Runnable on QEMU ARM

#![no_main]
#![no_std]

use cortex_m_rt::entry;
use cortex_m_semihosting::debug;
use cortex_m_semihosting::hprintln;
use panic_semihosting as _;

//

use log::{Record, Level, Metadata};
use log::LevelFilter;

struct HPrintLogger;

impl log::Log for HPrintLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Debug // compile-time log level
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            hprintln!("[{}] {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}

static LOGGER: HPrintLogger = HPrintLogger;

//

#[entry]
fn main() -> !
{
    log::set_logger(&LOGGER)
        .map(|()| log::set_max_level(LevelFilter::Debug)) // run-time log level
        .unwrap();
    hprintln!("run");

    test_lp();
    
    hprintln!("exit");
    debug::exit(debug::EXIT_SUCCESS);
    loop {}
}

//

use totsu::prelude::*;
use num_traits::Float;

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
        p.log_period = 10;
    });

    hprintln!("query_worklen -> {}", ASolver::query_worklen(op_a.size()));
    let mut solver_w = [0.; 48];

    let rslt = s.solve((op_c, op_a, op_b, cone, &mut solver_w)).unwrap();
    hprintln!("solve -> {:?}", rslt.0);

    assert!((rslt.0[0] - 2.).abs() <= 1e-3);
    assert!((rslt.0[1] - 2.).abs() <= 1e-3);
}
