extern crate intel_mkl_src;

use float_eq::assert_float_eq;
use totsu::prelude::*;
use totsu::linalg::F64LAPACK;
use totsu::operator::MatBuild;
use totsu::logger::PrintLogger;
use totsu::problem::ProbQP;

//

fn subtest_qp1<L: LinAlgEx<f64>>()
{
    let n = 2; // x0, x1
    let m = 1;
    let p = 0;
    
    // (1/2)(x - a)^2 + const
    let mut sym_p = MatBuild::new(MatType::SymPack(n));
    sym_p[(0, 0)] = 1.;
    sym_p[(1, 1)] = 1.;

    let mut vec_q = MatBuild::new(MatType::General(n, 1));
    vec_q[(0, 0)] = -(-1.); // -a0
    vec_q[(1, 0)] = -(-2.); // -a1
    
    // 1 - x0/b0 - x1/b1 <= 0
    let mut mat_g = MatBuild::new(MatType::General(m, n));
    mat_g[(0, 0)] = -1. / 2.; // -1/b0
    mat_g[(0, 1)] = -1. / 3.; // -1/b1
    
    let mut vec_h = MatBuild::new(MatType::General(m, 1));
    vec_h[(0, 0)] = -1.;

    let mat_a = MatBuild::new(MatType::General(p, n));

    let vec_b = MatBuild::new(MatType::General(p, 1));

    let s = Solver::<L, _>::new().par(|p| {p.max_iter = Some(100_000)});
    println!("{:?}", s.par);
    let mut qp = ProbQP::<L, _>::new(sym_p, vec_q, mat_g, vec_h, mat_a, vec_b, s.par.eps_zero);
    let rslt = s.solve(qp.problem(), PrintLogger).unwrap();
    println!("{:?}", rslt);

    assert_float_eq!(rslt.0[0..2], [2., 0.].as_ref(), abs_all <= 1e-3);
}

#[test]
fn test_qp1_1()
{
    subtest_qp1::<FloatGeneric<f64>>();
}

#[test]
fn test_qp1_2()
{
    subtest_qp1::<F64LAPACK>();
}
