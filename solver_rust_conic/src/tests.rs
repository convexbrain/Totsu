extern crate intel_mkl_src;

// TODO: pub use, mod layout
use crate::solver::*;
use crate::matop::*;
use crate::cone::*;
use crate::linalg::*;
use crate::matbuild::*;
use crate::logger::*;

#[test]
fn test_smoke1() {
    use float_eq::assert_float_eq;

    let op_c = MatOp::new(MatType::General(1, 1), &[
        1.,
    ]);

    let array_a = &[
         0., -1.,
        -1., -3.,
    ];
    let mat_a = MatBuild::new(MatType::SymPack(2))
                .iter_rowmaj(array_a)
                .scale_nondiag(2_f64.sqrt())
                .reshape_colvec();
    let op_a = MatOp::from(&mat_a);

    let array_b = &[
        1.,  0.,
        0., 10.,
    ];
    let mat_b = MatBuild::new(MatType::SymPack(2))
                .iter_rowmaj(array_b)
                .scale_nondiag(2_f64.sqrt())
                .reshape_colvec();
    let op_b = MatOp::from(&mat_b);

    let mut cone_w = vec![0.; ConePSD::query_worklen(op_a.size().0)];
    let cone = ConePSD::new(&mut cone_w);

    //let mut stdout = std::io::stdout();
    //let log = IoLogger(&mut stdout);
    let log = NullLogger;

    let s = Solver::new(F64BLAS, log);
    println!("{:?}", s.par);
    let mut solver_w = vec![0.; solver_query_worklen(op_a.size())];
    let rslt = s.solve((op_c, op_a, op_b, cone, &mut solver_w)).unwrap();
    println!("{:?}", rslt);

    assert_float_eq!(rslt.0[0], -2., abs_all <= 1e-3);
}
