extern crate intel_mkl_src;

use totsu::solver::{Solver, SolverError};
use totsu::linalg::{FloatGeneric, F64LAPACK};
use totsu::operator::{MatType, MatBuild};
use totsu::logger::PrintLogger;
use totsu::problem::ProbLP;

//

#[test]
fn test_lp1()
{
    type LA = F64LAPACK;
    type _LA = FloatGeneric<f64>;
    type ASolver = Solver<LA, f64>;
    type AProbLP = ProbLP<LA, f64>;
    type AMatBuild = MatBuild<LA, f64>;

    let n = 1;
    let m = 2;
    let p = 0;

    let vec_c = AMatBuild::new(MatType::General(n, 1)).iter_colmaj(&[
        1.,
    ]);

    // x <= b, x >= c
    let mat_g = AMatBuild::new(MatType::General(m, n)).iter_rowmaj(&[
        1., -1.,
    ]);
    let vec_h = AMatBuild::new(MatType::General(m, 1)).iter_colmaj(&[
        -5., // b
        -(10.)  // -c
    ]);

    let mat_a = AMatBuild::new(MatType::General(p, n));

    let vec_b = AMatBuild::new(MatType::General(p, 1));


    let s = ASolver::new();
    println!("{:?}", s.par);
    let mut lp = AProbLP::new(vec_c, mat_g, vec_h, mat_a, vec_b);
    let rslt = s.solve(lp.problem(), PrintLogger).unwrap_err();
    println!("{:?}", rslt);
    
    assert_eq!(rslt, SolverError::Infeasible);
}
