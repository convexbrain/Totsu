extern crate intel_mkl_src;

// TODO: pub use, mod layout
use crate::solver::*;
use crate::matop::*;
use crate::cone::*;
use crate::f64lapack::*;
use crate::floatgeneric::*;
use crate::matbuild::*;
use crate::stdlogger::*;

// TODO: more tests

//

#[test]
fn test_smoke1() {
    use float_eq::assert_float_eq;

    type _LA = F64LAPACK;
    type LA = FloatGeneric<f64>;
    type AMatOp<'a> = MatOp<'a, LA, f64>;
    type AConePSD<'a> = ConePSD<'a, LA, f64>;
    type ASolver = Solver<LA, f64>;
    type AMatBuild = MatBuild<LA, f64>;

    let op_c = AMatOp::new(MatType::General(1, 1), &[
        1.,
    ]);

    let array_a = &[
         0., -1.,
        -1., -3.,
    ];
    let mat_a = AMatBuild::new(MatType::SymPack(2))
                .iter_rowmaj(array_a)
                .scale_nondiag(2_f64.sqrt())
                .reshape_colvec();
    let op_a = AMatOp::from(&mat_a);

    let array_b = &[
        1.,  0.,
        0., 10.,
    ];
    let mat_b = AMatBuild::new(MatType::SymPack(2))
                .iter_rowmaj(array_b)
                .scale_nondiag(2_f64.sqrt())
                .reshape_colvec();
    let op_b = AMatOp::from(&mat_b);

    let mut cone_w = vec![0.; AConePSD::query_worklen(op_a.size().0)];
    let cone = AConePSD::new(&mut cone_w);

    let s = ASolver::new();
    println!("{:?}", s.par);
    let mut solver_w = vec![0.; ASolver::query_worklen(op_a.size())];
    let rslt = s.solve((op_c, op_a, op_b, cone, &mut solver_w), PrintLogger).unwrap();
    println!("{:?}", rslt);

    assert_float_eq!(rslt.0[0], -2., abs_all <= 1e-3);
}
