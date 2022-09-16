use float_eq::assert_float_eq;
use totsu::prelude::*;
use totsu::*;

type La = FloatGeneric<f64>;

type AMatBuild = MatBuild<La>;
type AProbQP = ProbQP<La>;
type ASolver = Solver<La>;

//

#[test]
fn test_qp1()
{
    let _ = env_logger::builder().is_test(true).try_init();

    let n = 2; // x0, x1
    let m = 1;
    let p = 0;
    
    // (1/2)(x - a)^2 + const
    let mut sym_p = AMatBuild::new(MatType::SymPack(n));
    sym_p[(0, 0)] = 1.;
    sym_p[(1, 1)] = 1.;

    let mut vec_q = AMatBuild::new(MatType::General(n, 1));
    vec_q[(0, 0)] = -(-1.); // -a0
    vec_q[(1, 0)] = -(-2.); // -a1
    
    // 1 - x0/b0 - x1/b1 <= 0
    let mut mat_g = AMatBuild::new(MatType::General(m, n));
    mat_g[(0, 0)] = -1. / 2.; // -1/b0
    mat_g[(0, 1)] = -1. / 3.; // -1/b1
    
    let mut vec_h = AMatBuild::new(MatType::General(m, 1));
    vec_h[(0, 0)] = -1.;

    let mat_a = AMatBuild::new(MatType::General(p, n));

    let vec_b = AMatBuild::new(MatType::General(p, 1));

    let s = ASolver::new().par(|p| {p.max_iter = Some(100_000)});
    let mut qp = AProbQP::new(sym_p, vec_q, mat_g, vec_h, mat_a, vec_b, s.par.eps_zero);
    let rslt = s.solve(qp.problem()).unwrap();
    println!("{:?}", rslt);

    assert_float_eq!(rslt.0[0..2], [2., 0.].as_ref(), abs_all <= 1e-3);
}
