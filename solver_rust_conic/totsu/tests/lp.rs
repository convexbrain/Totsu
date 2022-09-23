use totsu::prelude::*;
use totsu::*;

type La = FloatGeneric<f64>;

type AMatBuild = MatBuild<La>;
type AProbLP = ProbLP<La>;
type ASolver = Solver<La>;

//

#[test]
fn test_lp1()
{
    let _ = env_logger::builder().is_test(true).try_init();

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
          -5. ,  // b
        -(10.),  // -c
    ]);

    let mat_a = AMatBuild::new(MatType::General(p, n));

    let vec_b = AMatBuild::new(MatType::General(p, 1));


    let s = ASolver::new().par(|p| {p.max_iter = Some(100_000)});
    let mut lp = AProbLP::new(vec_c, mat_g, vec_h, mat_a, vec_b);
    let rslt = s.solve(lp.problem()).unwrap_err();
    println!("{}", rslt);
    
    assert_eq!(rslt, SolverError::Infeasible);
}

//

#[test]
fn test_lp2()
{
    let _ = env_logger::builder().is_test(true).try_init();

    let n = 1;
    let m = 2;
    let p = 0;

    let vec_c = AMatBuild::new(MatType::General(n, 1)).iter_colmaj(&[
        1.,
    ]);

    // x <= b, x <= c
    let mat_g = AMatBuild::new(MatType::General(m, n)).iter_rowmaj(&[
        1., 1.,
    ]);
    let vec_h = AMatBuild::new(MatType::General(m, 1)).iter_colmaj(&[
         5.,  // b
        10.,  // c
    ]);

    let mat_a = AMatBuild::new(MatType::General(p, n));

    let vec_b = AMatBuild::new(MatType::General(p, 1));


    let s = ASolver::new().par(|p| {p.max_iter = Some(100_000)});
    let mut lp = AProbLP::new(vec_c, mat_g, vec_h, mat_a, vec_b);
    let rslt = s.solve(lp.problem()).unwrap_err();
    println!("{}", rslt);
    
    assert_eq!(rslt, SolverError::Unbounded);
}
