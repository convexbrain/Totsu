use float_eq::assert_float_eq;
use totsu::prelude::*;
use totsu::operator::MatBuild;
use totsu::problem::ProbSOCP;

//

fn subtest_socp1<L: LinAlgEx<f64>>()
{
    let _ = env_logger::builder().is_test(true).try_init();

    let n = 2; // x0, x1
    let m = 1;
    let p = 0;
    let ni = 2;

    let mut vec_f = MatBuild::new(MatType::General(n, 1));
    vec_f.set_by_fn(|_, _| {1.});

    let mut mats_g = vec![MatBuild::new(MatType::General(ni, n)); m];
    mats_g[0][(0, 0)] = 1.;
    mats_g[0][(1, 1)] = 1.;

    let vecs_h = vec![MatBuild::new(MatType::General(ni, 1)); m];

    let vecs_c = vec![MatBuild::new(MatType::General(n, 1)); m];

    let mut scls_d = vec![0.; m];
    scls_d[0] = 2_f64.sqrt();

    let mat_a = MatBuild::new(MatType::General(p, n));

    let vec_b = MatBuild::new(MatType::General(p, 1));

    let s = Solver::<L, _>::new();
    println!("{:?}", s.par);
    let mut socp = ProbSOCP::<L, _>::new(vec_f, mats_g, vecs_h, vecs_c, scls_d, mat_a, vec_b);
    let rslt = s.solve(socp.problem()).unwrap();
    println!("{:?}", rslt);

    assert_float_eq!(rslt.0, [-1., -1.].as_ref(), abs_all <= 1e-3);
}

#[test]
fn test_socp1()
{
    subtest_socp1::<FloatGeneric<f64>>();
}

//

fn subtest_socp2<L: LinAlgEx<f64>>()
{
    let _ = env_logger::builder().is_test(true).try_init();

    // minimize f
    // 0 <= -f + 50
    // |-x+2| <= f
    // expected x=2, f=0

    let n = 2;
    let m = 2;
    let p = 0;

    let vec_f = MatBuild::new(MatType::General(n, 1)).iter_colmaj(&[0., 1.]);

    let mats_g = vec![
        MatBuild::new(MatType::General(0, n)),
        MatBuild::new(MatType::General(1, n)).iter_rowmaj(&[-1.0, 0.0]),
    ];

    let vecs_h = vec![
        MatBuild::new(MatType::General(0, 1)),
        MatBuild::new(MatType::General(1, 1)).iter_colmaj(&[2.]),
    ];

    let vecs_c = vec![
        MatBuild::new(MatType::General(m, 1)).iter_colmaj(&[0., -1.0]),
        MatBuild::new(MatType::General(m, 1)).iter_colmaj(&[0., 1.0]),
    ];

    let scls_d = vec![50., 0.];

    let mat_a = MatBuild::new(MatType::General(p, n));

    let vec_b = MatBuild::new(MatType::General(p, 1));

    let s = Solver::<L, _>::new().par(|p| {p.max_iter = Some(100_000)});
    println!("{:?}", s.par);
    let mut socp = ProbSOCP::<L, _>::new(vec_f, mats_g, vecs_h, vecs_c, scls_d, mat_a, vec_b);
    let rslt = s.solve(socp.problem()).unwrap();
    println!("{:?}", rslt);

    assert_float_eq!(rslt.0, [2., 0.].as_ref(), abs_all <= 1e-3);
}

#[test]
fn test_socp2()
{
    subtest_socp2::<FloatGeneric<f64>>();
}

#[cfg(feature = "f64lapack")]
mod f64lapack
{
    use intel_mkl_src as _;
    use totsu::linalg::F64LAPACK;
    use super::*;

    #[test]
    fn test_socp1()
    {
        subtest_socp1::<F64LAPACK>();
    }
    
    #[test]
    fn test_socp2()
    {
        subtest_socp2::<F64LAPACK>();
    }
}
