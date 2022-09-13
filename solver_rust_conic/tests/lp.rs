use totsu::prelude::*;
use totsu::operator::MatBuild;
use totsu::problem::ProbLP;

//

fn subtest_lp1<L: LinAlgEx<F=f64>>()
{
    let _ = env_logger::builder().is_test(true).try_init();

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
          -5. ,  // b
        -(10.),  // -c
    ]);

    let mat_a = MatBuild::new(MatType::General(p, n));

    let vec_b = MatBuild::new(MatType::General(p, 1));


    let s = Solver::<L>::new().par(|p| {p.max_iter = Some(100_000)});
    let mut lp = ProbLP::new(vec_c, mat_g, vec_h, mat_a, vec_b);
    let rslt = s.solve(lp.problem()).unwrap_err();
    println!("{}", rslt);
    
    assert_eq!(rslt, SolverError::Infeasible);
}

#[test]
fn test_lp1()
{
    subtest_lp1::<FloatGeneric<f64>>();
}

//

fn subtest_lp2<L: LinAlgEx<F=f64>>()
{
    let _ = env_logger::builder().is_test(true).try_init();

    let n = 1;
    let m = 2;
    let p = 0;

    let vec_c = MatBuild::new(MatType::General(n, 1)).iter_colmaj(&[
        1.,
    ]);

    // x <= b, x <= c
    let mat_g = MatBuild::new(MatType::General(m, n)).iter_rowmaj(&[
        1., 1.,
    ]);
    let vec_h = MatBuild::new(MatType::General(m, 1)).iter_colmaj(&[
         5.,  // b
        10.,  // c
    ]);

    let mat_a = MatBuild::new(MatType::General(p, n));

    let vec_b = MatBuild::new(MatType::General(p, 1));


    let s = Solver::<L>::new().par(|p| {p.max_iter = Some(100_000)});
    let mut lp = ProbLP::new(vec_c, mat_g, vec_h, mat_a, vec_b);
    let rslt = s.solve(lp.problem()).unwrap_err();
    println!("{}", rslt);
    
    assert_eq!(rslt, SolverError::Unbounded);
}

#[test]
fn test_lp2()
{
    subtest_lp2::<FloatGeneric<f64>>();
}

#[cfg(feature = "f64lapack")]
mod f64lapack
{
    use intel_mkl_src as _;
    use totsu::linalg::F64LAPACK;
    use super::*;

    #[test]
    fn test_lp1()
    {
        subtest_lp1::<F64LAPACK>();
    }
    
    #[test]
    fn test_lp2()
    {
        subtest_lp2::<F64LAPACK>();
    }
}
