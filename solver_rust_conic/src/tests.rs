extern crate intel_mkl_src;

use crate::*;

#[test]
fn test_smoke() {
    let op_c = MatOp::new((1, 1), &[
        1.,
    ]);
    let array_a = &mut[
        0., -1.,
            -3.,
    ];
    let op_a = MatOp::new_sym((3, 1), array_a);
    let array_b = &mut[
        1.,  0.,
            10.,
    ];
    let op_b = MatOp::new_sym((3, 1), array_b);

    let mut cone_w = vec![0.; ConePSD::query_worklen(op_a.size().0)];
    let cone = ConePSD::new(&mut cone_w);

    Solver::solve(SolverParam::default(), op_c, op_a, op_b, cone);
}
