extern crate intel_mkl_src;

// TODO: pub use, mod layout
use crate::solver::*;
use crate::matop::*;
use crate::cone::*;
use crate::logger::*;
use crate::linalg::*;

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

    let mut stdout = std::io::stdout();
    let log = IoLogger(&mut stdout);
    //let log = NullLogger;

    let s = Solver::new(F64BLAS, log);
    println!("{:?}", s.par);
    let mut solver_w = vec![0.; s.query_worklen(op_a.size())];
    let rslt = s.solve(op_c, op_a, op_b, cone, &mut solver_w);
    println!("{:?}", rslt);
}
