use float_eq::assert_float_eq;
use totsu::prelude::*;

//

#[test]
fn test_solver1()
{
    let _ = env_logger::builder().is_test(true).try_init();

    type LA = FloatGeneric<f64>;
    type AMatOp<'a> = MatOp<'a, LA>;
    type AConePSD<'a> = ConePSD<'a, LA>;
    type ASolver = Solver<LA>;

    let op_c = AMatOp::new(MatType::General(1, 1), &[
        1.,
    ]);

    /*
    This vector is a symmetric matrix of
       0., -1.,
      -1., -3.,
    packing the upper-triangle by columns,
    and non-diagonals are scaled to match the resulted matrix norm with the vector norm.
    */
    let op_a = AMatOp::new(MatType::General(3, 1), &[
         0., -1. * 1.41421356, -3.,
    ]);

    /*
    This vector is a symmetric matrix of
       1.,  0.,
       0., 10.,
    packing the upper-triangle by columns,
    and non-diagonals are scaled to match the resulted matrix norm with the vector norm.
    */
    let op_b = AMatOp::new(MatType::General(3, 1), &[
         1., 0. * 1.41421356, 10.,
    ]);

    let s = ASolver::new().par(|p| {p.max_iter = Some(100_000)});
    println!("{:?}", s.par);
    
    let mut cone_w = vec![0.; AConePSD::query_worklen(op_a.size().0)];
    let cone = AConePSD::new(&mut cone_w, s.par.eps_zero);

    let mut work = vec![0.; ASolver::query_worklen(op_a.size())];
    let mut sol_x = vec![0.; op_a.size().1];
    let mut sol_y = vec![0.; op_a.size().0];
    let rslt = s.solve((op_c, op_a, op_b, cone, &mut work), &mut sol_x, &mut sol_y).unwrap();
    println!("{:?}", rslt);

    assert_float_eq!(sol_x[0], -2., abs_all <= 1e-3);
}

#[cfg(feature = "f64lapack")]
mod f64lapack
{
    use intel_mkl_src as _; // workaround in case of link failure
}
