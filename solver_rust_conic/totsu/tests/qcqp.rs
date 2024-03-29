use float_eq::assert_float_eq;
use totsu::prelude::*;
use totsu::*;

type La = FloatGeneric<f64>;

type AMatBuild = MatBuild<La>;
type AProbQCQP = ProbQCQP<La>;
type ASolver = Solver<La>;

//

#[test]
fn test_qcqp1()
{
    let _ = env_logger::builder().is_test(true).try_init();

    let n = 2; // x0, x1
    let m = 1;
    let p = 0;
    
    // (1/2)(x - a)^2 + const
    let mut syms_p = vec![AMatBuild::new(MatType::SymPack(n)); m + 1];
    syms_p[0][(0, 0)] = 1.;
    syms_p[0][(1, 1)] = 1.;

    let mut vecs_q = vec![AMatBuild::new(MatType::General(n, 1)); m + 1];
    vecs_q[0][(0, 0)] = -(5.); // -a0
    vecs_q[0][(1, 0)] = -(4.); // -a1

    // 1 - x0/b0 - x1/b1 <= 0
    vecs_q[1][(0, 0)] = -1. / 2.; // -1/b0
    vecs_q[1][(1, 0)] = -1. / 3.; // -1/b1

    let mut scls_r = vec![0.; m + 1];
    scls_r[1] = 1.;
    
    let mat_a = AMatBuild::new(MatType::General(p, n));

    let vec_b = AMatBuild::new(MatType::General(p, 1));

    let s = ASolver::new().par(|p| {p.max_iter = Some(100_000)});
    let mut qp = AProbQCQP::new(syms_p, vecs_q, scls_r, mat_a, vec_b, s.par.eps_zero);
    let rslt = s.solve(qp.problem()).unwrap();
    println!("{:?}", rslt);

    assert_float_eq!(rslt.0[0..2], [5., 4.].as_ref(), abs_all <= 1e-3);
}
