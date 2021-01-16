extern crate intel_mkl_src;

use float_eq::assert_float_eq;
use totsu::prelude::*;
use totsu::linalg::F64LAPACK;
use totsu::operator::MatBuild;
use totsu::logger::PrintLogger;
use totsu::problem::ProbSOCP;

//

#[test]
fn test_socp1()
{
    type _LA = F64LAPACK;
    type LA = FloatGeneric<f64>;
    type ASolver = Solver<LA, f64>;
    type AProbSOCP = ProbSOCP<LA, f64>;
    type AMatBuild = MatBuild<LA, f64>;

    let n = 2; // x0, x1
    let m = 1;
    let p = 0;
    let ni = 2;

    let mut vec_f = AMatBuild::new(MatType::General(n, 1));
    vec_f.set_by_fn(|_, _| {1.});

    let mut mats_g = vec![AMatBuild::new(MatType::General(ni, n)); m];
    mats_g[0][(0, 0)] = 1.;
    mats_g[0][(1, 1)] = 1.;

    let vecs_h = vec![AMatBuild::new(MatType::General(ni, 1)); m];

    let vecs_c = vec![AMatBuild::new(MatType::General(n, 1)); m];

    let mut scls_d = vec![0.; m];
    scls_d[0] = 2_f64.sqrt();

    let mat_a = AMatBuild::new(MatType::General(p, n));

    let vec_b = AMatBuild::new(MatType::General(p, 1));

    let s = ASolver::new();
    println!("{:?}", s.par);
    let mut socp = AProbSOCP::new(vec_f, mats_g, vecs_h, vecs_c, scls_d, mat_a, vec_b);
    let rslt = s.solve(socp.problem(), PrintLogger).unwrap();
    println!("{:?}", rslt);

    assert_float_eq!(rslt.0, [-1., -1.].as_ref(), abs_all <= 1e-3);
}

//

#[test]
fn test_socp2()
{
    type _LA = F64LAPACK;
    type LA = FloatGeneric<f64>;
    type ASolver = Solver<LA, f64>;
    type AProbSOCP = ProbSOCP<LA, f64>;
    type AMatBuild = MatBuild<LA, f64>;

    // minimize f
    // 0 <= -f + 50
    // |-x+2| <= f
    // expected x=2, f=0

    let n = 2;
    let m = 2;
    let p = 0;

    let vec_f = AMatBuild::new(MatType::General(n, 1)).iter_colmaj(&[0., 1.]);

    let mats_g = vec![
        AMatBuild::new(MatType::General(0, n)),
        AMatBuild::new(MatType::General(1, n)).iter_rowmaj(&[-1.0, 0.0]),
    ];

    let vecs_h = vec![
        AMatBuild::new(MatType::General(0, 1)),
        AMatBuild::new(MatType::General(1, 1)).iter_colmaj(&[2.]),
    ];

    let vecs_c = vec![
        AMatBuild::new(MatType::General(m, 1)).iter_colmaj(&[0., -1.0]),
        AMatBuild::new(MatType::General(m, 1)).iter_colmaj(&[0., 1.0]),
    ];

    let scls_d = vec![50., 0.];

    let mat_a = AMatBuild::new(MatType::General(p, n));

    let vec_b = AMatBuild::new(MatType::General(p, 1));

    let s = ASolver::new();
    println!("{:?}", s.par);
    let mut socp = AProbSOCP::new(vec_f, mats_g, vecs_h, vecs_c, scls_d, mat_a, vec_b);
    let rslt = s.solve(socp.problem(), PrintLogger).unwrap();
    println!("{:?}", rslt);

    assert_float_eq!(rslt.0, [2., 0.].as_ref(), abs_all <= 1e-3);
}
