use totsu::prelude::*;
use totsu::operator::MatBuild;
use totsu::problem::ProbLP;

use utils;

use rand::prelude::*;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use anyhow::Result;
use plotters::prelude::*;

type AMatBuild = MatBuild<FloatGeneric<f64>, f64>;
type AProbLP = ProbLP<FloatGeneric<f64>, f64>;
type ASolver = Solver<FloatGeneric<f64>, f64>;

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
fn main() -> Result<()> {
    env_logger::init();

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
        p.eps_acc = 1e-3;
        utils::set_par_by_env(p);
    });
    let mut lp = AProbLP::new(vec_c, mat_g, vec_h, mat_a, vec_b);
    let rslt = s.solve(lp.problem())?;
    //println!("{:?}", rslt);

    let (_z, spl) = rslt.0.split_at(l);
    let (alpha, spl) = spl.split_at(l);
    let (_beta, spl) = spl.split_at(l);
    let (bias, _) = spl.split_at(1);

    //----- graph plot

    let area = SVGBackend::new("plot.svg", (480, 360)).into_drawing_area();
    area.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&area)
        .margin(30)
        .build_cartesian_3d(
            -0.1..1.1,
            -1.25..1.25,
            -0.1..1.1
        )?;

    chart.with_projection(|mut pb| {
        pb.yaw = 0.5;
        pb.pitch = 0.3;
        pb.scale = 0.8;
        pb.into_matrix()
    });

    chart.configure_axes()
        .x_labels(5 + 1/*adj*/)
        .y_labels(5)
        .z_labels(5 + 1/*adj*/)
        .draw()?;

    let grid = 20;
    chart.draw_series(
        SurfaceSeries::xoz(
            (0..grid).map(|f| f as f64 / (grid - 1) as f64),
            (0..grid).map(|f| f as f64 / (grid - 1) as f64),
            |x0, x1| {
                let xi = AMatBuild::new(MatType::General(2, 1))
                         .iter_colmaj(&[x0, x1]);
                
                wx(&x, alpha, &xi) + bias[0]
            }
        )
        .style(RGBColor(0, 127, 0).mix(0.1).filled()),
    )?;
    for x0 in 0 .. grid {
        chart.draw_series(LineSeries::new(
            (0..grid).map(|x1| {
                let x0 = x0 as f64 / (grid - 1) as f64;
                let x1 = x1 as f64 / (grid - 1) as f64;
                let xi = AMatBuild::new(MatType::General(2, 1))
                         .iter_colmaj(&[x0, x1]);

                (x0, wx(&x, alpha, &xi) + bias[0], x1)
            }), RGBColor(0, 127, 0).mix(0.5),
        ))?;
    }
    for x1 in 0 .. grid {
        chart.draw_series(LineSeries::new(
            (0..grid).map(|x0| {
                let x0 = x0 as f64 / (grid - 1) as f64;
                let x1 = x1 as f64 / (grid - 1) as f64;
                let xi = AMatBuild::new(MatType::General(2, 1))
                         .iter_colmaj(&[x0, x1]);

                (x0, wx(&x, alpha, &xi) + bias[0], x1)
            }), RGBColor(0, 127, 0).mix(0.5),
        ))?;
    }

    for smp in 0 .. l {
        chart.draw_series(PointSeries::of_element(
            [(x[(0, smp)], y[(smp, 0)], x[(1, smp)])],
            if alpha[smp].abs() > 0.001 {5} else {2},
            RED.mix(0.8).filled(),
            &|coord, size, style| {
                EmptyElement::at(coord)
                    + Circle::new((0, 0), size, style)
            },
        ))?;
    }

    Ok(())
}
