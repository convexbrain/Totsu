use totsu::prelude::*;
use totsu::operator::MatBuild;
use totsu::linalg::F64LAPACK;
use totsu::problem::ProbSDP;

use utils;

use rand::prelude::*;
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256StarStar;
use plotters::prelude::*;
use intel_mkl_src as _;
use anyhow::Result;

type AMatBuild = MatBuild<F64LAPACK, f64>;
type AProbSDP = ProbSDP<F64LAPACK, f64>;
type ASolver = Solver<F64LAPACK, f64>;

const EPS_ACC: f64 = 1e-3;
const EPS_ZERO: f64 = 1e-12;

fn make_adj_matrix(x_num: usize, y_num: usize, seed: u64) -> AMatBuild
{
    let l = x_num * y_num; // # of nodes
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);

    //----- make adjacent matrix

    let mut sym_w = AMatBuild::new(MatType::SymPack(l));
    for i in 0..l {
        let x = i / y_num;
        let y = i % y_num;

        if x < x_num - 1 {
            sym_w[(i, i + y_num)] = rng.sample(StandardNormal); // positive: dissimilar, negative: similar
        }
        if y < y_num - 1 {
            sym_w[(i, i + 1)] = rng.sample(StandardNormal); // positive: dissimilar, negative: similar
        }
    }
    //println!("{:?}", sym_w);

    sym_w
}

fn make_sdp(sym_w: &AMatBuild) -> AProbSDP
{
    let l = sym_w.size().0; // # of nodes

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

    AProbSDP::new(vec_c, syms_f, mat_a, vec_b, EPS_ZERO)
}

fn sample_feasible(rslt_x: &[f64], sym_w: &AMatBuild, seed: u64) -> (f64, AMatBuild)
{
    let l = sym_w.size().0; // # of nodes
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);

    //----- random sampling to find the best feasible point

    let mut sym_x = AMatBuild::new(MatType::SymPack(l));
    let mut kk = 0;
    for j in 0..l {
        for i in 0..=j {
            sym_x[(i, j)] = rslt_x[kk];
            kk += 1;
        }
    }
    //println!("{:?}", sym_x);
    sym_x.set_sqrt(EPS_ZERO);
    //println!("{:?}", sym_x);

    let mut tmpx = AMatBuild::new(MatType::General(l, 1));
    let mut x = AMatBuild::new(MatType::General(l, 1));
    let mut o = [0f64; 1];
    let mut o_feas = None;
    let mut x_feas = AMatBuild::new(MatType::General(l, 1));

    for _ in 0..l {
        tmpx.set_by_fn(|_, _| rng.sample(StandardNormal));
        //println!("{:?}", tmpx);
    
        sym_x.op(1., tmpx.as_ref(), 0., x.as_mut());
        for e in x.as_mut() {
            *e = if *e > 0. {1.} else {-1.};
        }
        //println!("{:?}", x);

        sym_w.op(1., x.as_ref(), 0., tmpx.as_mut());
        tmpx.trans_op(1., x.as_ref(), 0., &mut o);
        //println!("{}", o[0]);

        if let Some(o_feas_some) = o_feas {
            if o_feas_some > o[0] {
                o_feas = Some(o[0]);
                x_feas.set_iter_colmaj(x.as_ref());
            }
        }
        else {
            o_feas = Some(o[0]);
            x_feas.set_iter_colmaj(x.as_ref());
        }
    }
    let o_feas = o_feas.unwrap();
    //println!("{}", o_feas);
    //println!("{:?}", x_feas);    

    (o_feas, x_feas)
}

/// main
fn main() -> Result<()> {
    env_logger::init();

    //----- make adjacent matrix

    let x_num = 8;
    let y_num = 6;
    let sym_w = make_adj_matrix(x_num, y_num, 10000);

    //----- formulate partitioning problem as SDP

    let mut sdp = make_sdp(&sym_w);

    //----- solve SDP

    let s = ASolver::new().par(|p| {
        p.eps_acc = EPS_ACC;
        p.eps_zero = EPS_ZERO;
        utils::set_par_by_env(p);
    });
    let rslt = s.solve(sdp.problem())?;
    //println!("{:?}", rslt);

    //----- random sampling to find the best feasible point

    let (_, x_feas) = sample_feasible(&rslt.0, &sym_w, 20000);
    //println!("{:?}", x_feas);

    //----- visualize

    let root = SVGBackend::new("plot.svg", (480, 360)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(30)
        .build_cartesian_2d(
            0..x_num,
            0..y_num
        )?;

    let scale = 5.;
    for i in 0..sym_w.size().0 {
        let x = i / y_num;
        let y = i % y_num;

        if x < x_num - 1 {
            let a = sym_w[(i, i + y_num)];

            let style = if a > 0. {RED} else {BLUE}.stroke_width((a.abs() * scale) as u32);

            chart.draw_series(LineSeries::new(
                [(x, y), (x + 1, y)], style
            ))?;
        }
        if y < y_num - 1 {
            let a = sym_w[(i, i + 1)];

            let style = if a > 0. {RED} else {BLUE}.stroke_width((a.abs() * scale) as u32);

            chart.draw_series(LineSeries::new(
                [(x, y), (x, y + 1)], style
            ))?;
        }
    }

    let radius = 10;
    for i in 0..sym_w.size().0 {
        let x = i / y_num;
        let y = i % y_num;

        if x_feas[(i, 0)] > 0. {
            chart.draw_series(
                [Circle::new((x, y), radius, BLACK.filled())]
            )?;
        } else {
            chart.draw_series(
                [Circle::new((x, y), radius, WHITE.filled()), Circle::new((x, y), radius, &BLACK)]
            )?;
        }

    }

    Ok(())
}
