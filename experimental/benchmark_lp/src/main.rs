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
    let m = n + sz;
    let p = 0;

    let mut rng = Xoshiro256StarStar::seed_from_u64(0);

    //----- construct LP

    let vec_c = MatBuild::new(MatType::General(n, 1))
                .by_fn(|_, _| {
                    -rng.gen::<f32>()
                });
    //println!("vec_c:\n{}", vec_c);

    let mat_g = MatBuild::new(MatType::General(m, n))
                .by_fn(|r, c| {
                    if r < n {
                        if r == c {
                            -1.
                        }
                        else {
                            0.
                        }
                    }
                    else {
                        rng.gen()
                    }
                });
    //println!("mat_g:\n{}", mat_g);

    let vec_h = MatBuild::new(MatType::General(m, 1))
                .by_fn(|r, _| {
                    if r < n {
                        0.
                    }
                    else {
                        rng.gen()
                    }
                });
    //println!("vec_h:\n{}", vec_h);
        
    let mat_a = MatBuild::new(MatType::General(p, n));
    let vec_b = MatBuild::new(MatType::General(p, 1));
    //println!("mat_a:\n{}", mat_a);
    //println!("vec_b:\n{}", vec_b);

    //----- solve LP

    let s = Solver::<L>::new()
            .par(|p| {
                p.eps_acc = 1e-3;
                p.log_period = 1000;
            });
    let mut lp = ProbLP::new(vec_c, mat_g, vec_h, mat_a, vec_b);
    let prob = lp.problem();
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

    bench::<FloatGeneric<f32>>(sz);
    bench::<F32CUDA>(sz);

    Ok(())
}
