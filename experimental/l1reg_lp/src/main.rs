use totsu::prelude::*;
use totsu::operator::MatBuild;
use totsu::linalg::F64LAPACK;
use totsu::logger::PrintLogger;
use totsu::problem::ProbLP;

use rand::prelude::*;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use std::io::prelude::*;
use std::io::BufWriter;
use std::fs::File;

extern crate intel_mkl_src;

type AMatBuild = MatBuild<F64LAPACK, f64>;
type AProbLP = ProbLP<F64LAPACK, f64>;
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
fn wx(x: &AMatBuild, alpha: &[f64], xi: &AMatBuild) -> f64
{
    assert_eq!(x.size().1, alpha.len());
    assert_eq!(xi.size().1, 1);

    let mut f = 0.;
    for i in 0 .. alpha.len() {
        f += alpha[i] * kernel(x, i, xi, 0);
    }
    f
}

/// main
fn main() -> std::io::Result<()> {

    //----- make sample points for training

    let mut rng = Xoshiro256StarStar::seed_from_u64(0);
    let l = 20; // # of samples
    let x = AMatBuild::new(MatType::General(2, l))
            .by_fn(|_, _| rng.gen()); // random 2-dimensional points
    let y = AMatBuild::new(MatType::General(l, 1))
            .by_fn(|smp, _| {
                let x0 = (5. * x[(0, smp)]).cos();
                let x1 = (7. * x[(1, smp)]).cos();
                x0 * x1 // wavy-shaped points
            });
    //println!("{}", x);
    //println!("{}", y);

    //----- formulate L1-regularized L1-error regression as LP

    let n = l * 3 + 1; // z, alpha, beta, bias
    let m = l * 4;
    let p = 0;

    let lambda = 0.2; // L1-regularization strength

    let mut vec_c = AMatBuild::new(MatType::General(n, 1));
    for i in 0.. l {
        vec_c[(i, 0)] = 1.; // for z, L1-norm error
        vec_c[(l * 2 + i, 0)] = lambda; // for beta, L1-regularization
    }
    //println!("{}", vec_c);

    let mut mat_g = AMatBuild::new(MatType::General(m, n));
    for i in 0.. l {
        mat_g[(i, i)] = -1.;
        mat_g[(l + i, i)] = -1.;

        mat_g[(l * 2 + i, l + i)] = 1.;
        mat_g[(l * 3 + i, l + i)] = -1.;

        mat_g[(l * 2 + i, l * 2 + i)] = -1.;
        mat_g[(l * 3 + i, l * 2 + i)] = -1.;

        mat_g[(i, l * 3)] = 1.;
        mat_g[(l + i, l * 3)] = -1.;
    }
    for (r, c) in itertools::iproduct!(0.. l, 0.. l) {
        mat_g[(r, l + c)] = kernel(&x, r, &x, c);
        mat_g[(l + r, l + c)] = -kernel(&x, r, &x, c);
    }
    //println!("{}", mat_g);

    let mut vec_h = AMatBuild::new(MatType::General(m, 1));
    for i in 0.. l {
        vec_h[(i, 0)] = y[(i, 0)];
        vec_h[(l + i, 0)] = -y[(i, 0)];
    }
    //println!("{}", vec_h);
        
    let mat_a = AMatBuild::new(MatType::General(p, n));
    let vec_b = AMatBuild::new(MatType::General(p, 1));
    //println!("{}", mat_a);
    //println!("{}", vec_b);

    //----- solve LP

    let s = ASolver::new().par(|p| {
        p.log_period = 10000;
    });
    let mut lp = AProbLP::new(vec_c, mat_g, vec_h, mat_a, vec_b);
    let rslt = s.solve(lp.problem(), PrintLogger).unwrap();
    //println!("{:?}", rslt);

    let (_z, spl) = rslt.0.split_at(l);
    let (alpha, spl) = spl.split_at(l);
    let (_beta, spl) = spl.split_at(l);
    let (bias, _) = spl.split_at(1);

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

            let f = wx(&x, alpha, &xi) + bias[0];

            write!(dat_grid, " {}", f)?;
        }
        writeln!(dat_grid)?;
    }

    Ok(())
}
