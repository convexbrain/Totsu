use totsu::prelude::*;
use totsu::operator::MatBuild;
use totsu::linalg::F64LAPACK;
use totsu::logger::PrintLogger;
use totsu::problem::ProbQP;

use rand::prelude::*;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

extern crate intel_mkl_src;

type AMatBuild = MatBuild<F64LAPACK, f64>;
type AProbQP = ProbQP<F64LAPACK, f64>;
type ASolver = Solver<F64LAPACK, f64>;

/// main
fn main() -> std::io::Result<()> {

    //----- make sample points for training

    let mut rng = Xoshiro256StarStar::seed_from_u64(0);
    let l = 50; // # of dimension

    let x = AMatBuild::new(MatType::General(l, 1))
            .by_fn(|_, _| rng.gen_range(-1. ..= 1.)); // random l-dimensional vector
    //println!("{}", x);

    let mut y = vec![AMatBuild::new(MatType::General(l, 1)); l];
    for yi in y.iter_mut() {
        yi.set_by_fn(|_, _| rng.gen_range(-1. ..= 1.)); // random l-dimensional vector
        //println!("{}", yi);
    }

    //----- formulate least square as QP

    let n = l;
    let m = 0;
    let p = 0;

    let sym_p = AMatBuild::new(MatType::SymPack(n))
                .by_fn(|r, c| {
                    let mut sum = 0.;
                    for (yr, yc) in y[r].as_ref().iter().zip(y[c].as_ref()) {
                        sum += yr * yc;
                    }
                    sum
                });
    //println!("{}", sym_p);

    let mut vec_q = x.clone();
    sym_p.op(-1., x.as_ref(), 0., vec_q.as_mut());
    //println!("{}", vec_q);

    let mat_g = AMatBuild::new(MatType::General(m, n));
    //println!("{}", mat_g);

    let vec_h = AMatBuild::new(MatType::General(m, 1));
    //println!("{}", vec_h);

    let mat_a = AMatBuild::new(MatType::General(p, n));
    //println!("{}", mat_a);

    let vec_b = AMatBuild::new(MatType::General(p, 1));
    //println!("{}", vec_b);

    //----- solve QP

    let s = ASolver::new().par(|p| {
        p.log_period = Some(10000);
    });
    let mut qp = AProbQP::new(sym_p, vec_q, mat_g, vec_h, mat_a, vec_b, s.par.eps_zero);
    let rslt = s.solve(qp.problem(), PrintLogger).unwrap();
    //println!("{:?}", rslt.0);

    let mut max_absdiff = 0_f64;
    for i in 0.. l {
        let absdiff = (x[(i, 0)] - rslt.0[i]).abs();
        max_absdiff = max_absdiff.max(absdiff);
    }
    println!("max_absdiff {:.3e}", max_absdiff);

    //println!("{:?}", x);

    Ok(())
}
