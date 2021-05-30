use float_eq::assert_float_eq;
use totsu::prelude::*;
use totsu::operator::MatBuild;
use totsu::problem::ProbSDP;

//

fn subtest_sdp1<L: LinAlgEx<f64>>()
{
    let _ = env_logger::builder().is_test(true).try_init();

    let n = 2;
    let p = 0;
    let k = 2;

    let vec_c = MatBuild::new(MatType::General(n, 1)).iter_colmaj(&[
        1., 1.,
    ]);

    let mut syms_f = vec![MatBuild::new(MatType::SymPack(k)); n + 1];

    syms_f[0].set_iter_rowmaj(&[
        -1., 0.,
         0., 0.,
    ]);
    syms_f[1].set_iter_rowmaj(&[
        0.,  0.,
        0., -1.,
    ]);
    syms_f[2].set_iter_rowmaj(&[
        3., 0.,
        0., 4.,
    ]);

    let mat_a = MatBuild::new(MatType::General(p, n));

    let vec_b = MatBuild::new(MatType::General(p, 1));

    let s = Solver::<L, _>::new().par(|p| {p.max_iter = Some(100_000)});
    println!("{:?}", s.par);
    let mut sdp = ProbSDP::<L, _>::new(vec_c, syms_f, mat_a, vec_b, s.par.eps_zero);
    let rslt = s.solve(sdp.problem()).unwrap();
    println!("{:?}", rslt);

    assert_float_eq!(rslt.0, [3., 4.].as_ref(), abs_all <= 1e-3);
}

#[test]
fn test_sdp1()
{
    subtest_sdp1::<FloatGeneric<f64>>();
}

#[cfg(feature = "f64lapack")]
mod f64lapack
{
    use intel_mkl_src as _;
    use totsu::linalg::F64LAPACK;
    use super::*;

    #[test]
    fn test_sdp1()
    {
        subtest_sdp1::<F64LAPACK>();
    }
}
