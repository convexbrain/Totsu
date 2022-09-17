use float_eq::assert_float_eq;
use totsu::prelude::*;
use totsu::*;

type La = FloatGeneric<f64>;

type AMatBuild = MatBuild<La>;
type AProbSDP = ProbSDP<La>;
type ASolver = Solver<La>;

//

#[test]
fn test_sdp1()
{
    let _ = env_logger::builder().is_test(true).try_init();

    let n = 2;
    let p = 0;
    let k = 2;

    let vec_c = AMatBuild::new(MatType::General(n, 1)).iter_colmaj(&[
        1., 1.,
    ]);

    let mut syms_f = vec![AMatBuild::new(MatType::SymPack(k)); n + 1];

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

    let mat_a = AMatBuild::new(MatType::General(p, n));

    let vec_b = AMatBuild::new(MatType::General(p, 1));

    let s = ASolver::new().par(|p| {p.max_iter = Some(100_000)});
    let mut sdp = AProbSDP::new(vec_c, syms_f, mat_a, vec_b, s.par.eps_zero);
    let rslt = s.solve(sdp.problem()).unwrap();
    println!("{:?}", rslt);

    assert_float_eq!(rslt.0, [3., 4.].as_ref(), abs_all <= 1e-3);
}
