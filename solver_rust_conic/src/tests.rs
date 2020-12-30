extern crate intel_mkl_src;

use crate::prelude::*;
use crate::predef::*;

#[test]
fn test_smoke() {
    let s = Solver::new();

    let op_c = MatOp::new((1, 1), &[
        1.,
    ]);
    let op_a = MatOp::new((3, 1), &[
        0.,
        -1. * 1.41421356,
        -3.,
    ]);
    let op_b = MatOp::new((3, 1), &[
        1.,
        0. * 1.41421356,
        10.,
    ]);

    s.solve(SolverParam::default(), op_c, op_a, op_b);
}
