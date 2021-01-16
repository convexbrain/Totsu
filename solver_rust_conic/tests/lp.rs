extern crate intel_mkl_src;

use totsu::prelude::*;
use totsu::linalg::F64LAPACK;
use totsu::operator::MatBuild;
use totsu::logger::PrintLogger;
use totsu::problem::ProbLP;

//

fn subtest_lp1<L: LinAlgEx<f64>>()
{
    let n = 1;
    let m = 2;
    let p = 0;

    let vec_c = MatBuild::new(MatType::General(n, 1)).iter_colmaj(&[
        1.,
    ]);

    // x <= b, x >= c
    let mat_g = MatBuild::new(MatType::General(m, n)).iter_rowmaj(&[
        1., -1.,
    ]);
    let vec_h = MatBuild::new(MatType::General(m, 1)).iter_colmaj(&[
        -5., // b
        -(10.)  // -c
    ]);

    let mat_a = MatBuild::new(MatType::General(p, n));

    let vec_b = MatBuild::new(MatType::General(p, 1));


    let s = Solver::<L, _>::new();
    println!("{:?}", s.par);
    let mut lp = ProbLP::<L, _>::new(vec_c, mat_g, vec_h, mat_a, vec_b);
    let rslt = s.solve(lp.problem(), PrintLogger).unwrap_err();
    println!("{:?}", rslt);
    
    assert_eq!(rslt, SolverError::Infeasible);
}

#[test]
fn test_lp1_1()
{
    subtest_lp1::<FloatGeneric<f64>>();
}

#[test]
fn test_lp1_2()
{
    subtest_lp1::<F64LAPACK>();
}
