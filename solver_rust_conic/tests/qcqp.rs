use float_eq::assert_float_eq;
use totsu::prelude::*;
use totsu::operator::MatBuild;
use totsu::logger::PrintLogger;
use totsu::problem::ProbQCQP;

//

fn subtest_qcqp1<L: LinAlgEx<f64>>()
{
    let n = 2; // x0, x1
    let m = 1;
    let p = 0;
    
    // (1/2)(x - a)^2 + const
    let mut syms_p = vec![MatBuild::new(MatType::SymPack(n)); m + 1];
    syms_p[0][(0, 0)] = 1.;
    syms_p[0][(1, 1)] = 1.;

    let mut vecs_q = vec![MatBuild::new(MatType::General(n, 1)); m + 1];
    vecs_q[0][(0, 0)] = -(5.); // -a0
    vecs_q[0][(1, 0)] = -(4.); // -a1

    // 1 - x0/b0 - x1/b1 <= 0
    vecs_q[1][(0, 0)] = -1. / 2.; // -1/b0
    vecs_q[1][(1, 0)] = -1. / 3.; // -1/b1

    let mut scls_r = vec![0.; m + 1];
    scls_r[1] = 1.;
    
    let mat_a = MatBuild::new(MatType::General(p, n));

    let vec_b = MatBuild::new(MatType::General(p, 1));

    let s = Solver::<L, _>::new().par(|p| {p.max_iter = Some(100_000)});
    println!("{:?}", s.par);
    let mut qp = ProbQCQP::<L, _>::new(syms_p, vecs_q, scls_r, mat_a, vec_b, s.par.eps_zero);
    let rslt = s.solve(qp.problem(), PrintLogger).unwrap();
    println!("{:?}", rslt);

    assert_float_eq!(rslt.0[0..2], [5., 4.].as_ref(), abs_all <= 1e-3);
}

#[test]
fn test_qcqp1()
{
    subtest_qcqp1::<FloatGeneric<f64>>();
}

#[cfg(feature = "f64lapack")]
mod f64lapack
{
    use intel_mkl_src as _;
    use totsu::linalg::F64LAPACK;
    use super::*;

    #[test]
    fn test_qcqp1()
    {
        subtest_qcqp1::<F64LAPACK>();
    }
}
