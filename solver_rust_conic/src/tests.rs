extern crate intel_mkl_src;

use crate::prelude::*;
use crate::predef::*;

#[test]
fn test_smoke() {
    let op_c = MatOp::new((1, 1), &[
        1.,
    ]);
    // TODO: mat_to_vec
    let op_a = MatOp::new((3, 1), &[
        0.,
        -1. * 1.41421356,
        -3.,
    ]);
    // TODO: mat_to_vec
    let op_b = MatOp::new((3, 1), &[
        1.,
        0. * 1.41421356,
        10.,
    ]);

    let mut proj_w = vec![0.; ProjPSD::query_worklen(op_a.size().0)];
    let proj = ProjPSD::new(&mut proj_w);

    let s = Solver::new();
    s.solve(SolverParam::default(), op_c, op_a, op_b, proj);
}
