use totsu::prelude::*;
use totsu::*;
use totsu_f32cuda::F32CUDA;
use totsu_core::LinAlgEx;

use rand::prelude::*;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

use std::str::FromStr;


fn bench<L: LinAlgEx<F=f32>>(sz: usize) {
    let n = sz;
    let m = sz;
    let p = 0;

    let mut rng = Xoshiro256StarStar::seed_from_u64(0);

    //----- construct QP

    let sym_p = MatBuild::new(MatType::SymPack(n))
                .by_fn(|r, c| {
                    if r == c {
                        rng.gen()
                    }
                    else {
                        0.
                    }
                });
    //println!("sym_p:\n{}", sym_p);

    let vec_q = MatBuild::new(MatType::General(n, 1))
                .by_fn(|_, _| {
                    rng.gen()
                });
    //println!("vec_q:\n{}", vec_q);

    let mat_g = MatBuild::new(MatType::General(m, n))
                .by_fn(|_, _| {
                    -rng.gen::<f32>()
                });
    //println!("mat_g:\n{}", mat_g);

    let vec_h = MatBuild::new(MatType::General(m, 1))
                .by_fn(|_, _| {
                    -rng.gen::<f32>()
                });
    //println!("vec_h:\n{}", vec_h);
        
    let mat_a = MatBuild::new(MatType::General(p, n));
    let vec_b = MatBuild::new(MatType::General(p, 1));
    //println!("mat_a:\n{}", mat_a);
    //println!("vec_b:\n{}", vec_b);

    //----- solve QP

    let s = Solver::<L>::new()
            .par(|p| {
                p.eps_acc = 1e-3;
            });
    let mut qp = ProbQP::new(sym_p, vec_q, mat_g, vec_h, mat_a, vec_b, s.par.eps_zero);
    let prob = qp.problem();
    let rslt = s.solve(prob).unwrap();

    let vec_r = MatBuild::<L>::new(MatType::General(n, 1))
                   .iter_colmaj(rslt.0);

    println!("result:\n{}", vec_r);
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let mut sz = 100;
    
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 {
        if let Ok(a) = usize::from_str(&args[1]) {
            sz = a; // sz can be specified by 1st argument
        }
    }

    bench::<F32CUDA>(sz);
    bench::<FloatGeneric<f32>>(sz);

    Ok(())
}
