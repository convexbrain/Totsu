use totsu::prelude::*;
use totsu::operator::MatBuild;
use totsu::linalg::F64LAPACK;
use totsu::problem::ProbSDP;

use rand::prelude::*;
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256StarStar;

extern crate intel_mkl_src;

type AMatBuild = MatBuild<F64LAPACK, f64>;
type AProbSDP = ProbSDP<F64LAPACK, f64>;
type ASolver = Solver<F64LAPACK, f64>;

/// main
fn main() -> std::io::Result<()> {
    env_logger::init();

    //----- make adjacent matrix

    let l = 9; // # of nodes
    let mut sym_w = AMatBuild::new(MatType::SymPack(l));
    sym_w[(0, 3)] = 1.;
    sym_w[(1, 2)] = 1.;
    sym_w[(2, 5)] = 1.;
    sym_w[(3, 4)] = 1.;
    sym_w[(4, 5)] = 1.;
    sym_w[(4, 7)] = 1.;
    sym_w[(5, 8)] = 1.;
    sym_w[(6, 7)] = 1.;
    /*
    let l = 4; // # of nodes
    let mut sym_w = AMatBuild::new(MatType::SymPack(l));
    sym_w[(0, 1)] = 1.;
    sym_w[(0, 2)] = 1.;
    sym_w[(0, 3)] = 1.;
    */

    //----- formulate max-cut as SDP

    let vec_c = sym_w.clone().reshape_colvec();
    //println!("{}", vec_c);

    let n = vec_c.size().0;
    let k = l;
    let p = l;

    let mut syms_f = vec![AMatBuild::new(MatType::SymPack(k)); n + 1];
    let mut kk = 0;
    for j in 0..l {
        for i in 0..=j {
            syms_f[kk][(i, j)] = -1.;
            kk += 1;
        }
    }

    let mut mat_a = AMatBuild::new(MatType::General(p, n));
    let mut j = 0;
    for i in 0..p {
        mat_a[(i, j)] = 1.;
        j += i + 2;
    }
    //println!("{}", mat_a);

    let vec_b = AMatBuild::new(MatType::General(p, 1))
                .by_fn(|_, _| 1.);
    //println!("{}", vec_b);

    //----- solve SDP

    let s = ASolver::new().par(|p| {
        p.eps_acc = 1e-3;
    });
    let mut sdp = AProbSDP::new(vec_c, syms_f, mat_a, vec_b, s.par.eps_zero);
    let rslt = s.solve(sdp.problem()).unwrap();
    //println!("{:?}", rslt);

    //----- random sampling of feasible point

    let mut sym_x = AMatBuild::new(MatType::SymPack(l));
    let mut kk = 0;
    for j in 0..l {
        for i in 0..=j {
            sym_x[(i, j)] = rslt.0[kk];
            kk += 1;
        }
    }
    //println!("{:?}", sym_x);
    sym_x.set_sqrt(1e-3);
    //println!("{:?}", sym_x);

    let mut rng = Xoshiro256StarStar::seed_from_u64(10000);

    let mut nx = vec![0_f64; l];
    let mut x = vec![0_f64; l];
    for e in &mut nx {
        *e = rng.sample(StandardNormal);
    }
    //println!("{:?}", nx);

    sym_x.op(1., &nx, 0., &mut x);
    for e in &mut x {
        *e = if *e > 0. {1.} else {-1.};
    }
    println!("{:?}", x);

    Ok(())
}
