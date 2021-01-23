use totsu::prelude::*;
use totsu::operator::MatBuild;
use totsu::linalg::F64LAPACK;
use totsu::logger::PrintLogger;
use totsu::problem::ProbQP;

use rand::prelude::*;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use std::io::prelude::*;
use std::io::BufWriter;
use std::fs::File;

extern crate intel_mkl_src;

type AMatBuild = MatBuild<F64LAPACK, f64>;
type AProbQP = ProbQP<F64LAPACK, f64>;
type ASolver = Solver<F64LAPACK, f64>;

/// gaussian kernel
fn kernel(xi: &AMatBuild, ci: usize, xj: &AMatBuild, cj: usize) -> f64
{
    assert_eq!(xi.size().0, xj.size().0);
    assert!(ci < xi.size().1);
    assert!(cj < xj.size().1);

    let sigma_sq = 1. / 8.;
    let mut norm_sq = 0.;
    for r in 0.. xi.size().0 {
        let d = xi[(r, ci)] - xj[(r, cj)];
        norm_sq += d * d;
    }
    (-norm_sq / sigma_sq).exp()
}

/// kerneled w*x
fn wx(x: &AMatBuild, y: &AMatBuild, alpha: &[f64], xj: &AMatBuild, cj: usize) -> f64
{
    assert_eq!(y.size().0, alpha.len());
    assert_eq!(y.size().1, 1);

    let mut f = 0.;
    for i in 0 .. alpha.len() {
        f += alpha[i] * y[(i, 0)] * kernel(x, i, xj, cj);
    }
    f
}

/// main
fn main() -> std::io::Result<()> {

    //----- make sample points for training

    let mut rng = Xoshiro256StarStar::seed_from_u64(10);
    let l = 10; // TODO 50; // # of samples
    let x = AMatBuild::new(MatType::General(2, l))
            .by_fn(|_, _| rng.gen()); // random 2-dimensional points
    let y = AMatBuild::new(MatType::General(l, 1))
            .by_fn(|smp, _| {
                let (d0, d1) = (x[(0, smp)] - 0.5, x[(1, smp)] - 0.5);
                let r2 = d0 * d0 + d1 * d1;
                let r = r2.sqrt();
                if (r > 0.25) && (r < 0.4) {1.} else {-1.} // ring shape, 1 inside, -1 outside and hole
            });
    //println!("{}", x);
    //println!("{}", y);

    //----- formulate dual of hard-margin SVM as QP

    let n = l;
    let m = l;
    let p = 1;

    let sym_p = AMatBuild::new(MatType::SymPack(n))
                .by_fn(|r, c| {
                    y[(r, 0)] * y[(c, 0)] * kernel(&x, r, &x, c)
                });
    //println!("{}", sym_p);

    let vec_q = AMatBuild::new(MatType::General(n, 1))
                .by_fn(|_, _| -1.);
    //println!("{}", vec_q);

    let mut mat_g = AMatBuild::new(MatType::General(m, n));
    for i in 0.. m.min(n) {
        mat_g[(i, i)] = -1.;
    }
    //println!("{}", mat_g);

    let vec_h = AMatBuild::new(MatType::General(m, 1));
    //println!("{}", vec_h);

    let mat_a = AMatBuild::new(MatType::General(p, n))
                    .by_fn(|_, c| y[(c, 0)]);
    //println!("{}", mat_a);

    let vec_b = AMatBuild::new(MatType::General(p, 1));
    //println!("{}", vec_b);

    //----- solve QP

    let mut s = ASolver::new(); // TODO
    //s.par.eps_acc = 1e-3;
    s.par.log_period = 1000;
    let mut qp = AProbQP::new(sym_p, vec_q, mat_g, vec_h, mat_a, vec_b);
    let rslt = s.solve(qp.problem(), PrintLogger).unwrap();
    //println!("{:?}", rslt);

    let (alpha, _) = rslt.0.split_at(l);

    //----- calculate bias term

    let mut bias = 0.;
    let mut cnt = 0;
    for i in 0 .. l {
        if rslt.0[i] > 1e-4 {
            bias += y[(i, 0)] - wx(&x, &y, alpha, &x, i);
            cnt += 1;
        }
    }
    bias = bias / cnt as f64;

    //----- file output for graph plot
    // TODO: use some plotting crate

    let mut dat_point = BufWriter::new(File::create("dat_point")?);

    for smp in 0 .. l {
        writeln!(dat_point, "{} {} {} {}", x[(0, smp)], x[(1, smp)], y[(smp, 0)], alpha[smp])?;
    }

    let mut dat_grid = BufWriter::new(File::create("dat_grid")?);

    let grid = 20;
    for iy in 0 .. grid {
        for ix in 0 .. grid {
            let xi = AMatBuild::new(MatType::General(2, 1))
                     .iter_colmaj(&[
                         ix as f64 / (grid - 1) as f64,
                         iy as f64 / (grid - 1) as f64
                     ]);

            let f = wx(&x, &y, alpha, &xi, 0) + bias;

            write!(dat_grid, " {}", f)?;
        }
        writeln!(dat_grid)?;
    }

    Ok(())
}
