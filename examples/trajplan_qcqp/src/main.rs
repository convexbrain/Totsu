use totsu::prelude::*;
use totsu::*;
use totsu_f64lapack::F64LAPACK;

use std::str::FromStr;

use plotters::prelude::*;
use intel_mkl_src as _;
use anyhow::Result;

type La = F64LAPACK;
type AMatBuild = MatBuild<La>;
type AProbQCQP = ProbQCQP<La>;
type ASolver = Solver<La>;

/// main
fn main() -> Result<()> {
    env_logger::init();

    //----- parameters

    let t_cap = 30; // # of grids dividing time [0, 1)
    let mut a_cap = 90.0; // limit acceleration magnitude
    let x_s = (0., 0.); // start position
    let x_m1 = (0.5, -1.5); // target position at time=1/3
    let x_m2 = (0.25, 1.5); // target position at time=2/3
    let x_t = (1., 1.); // terminal position

    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 {
        if let Ok(a) = f64::from_str(&args[1]) {
            a_cap = a; // a_cap can be specified by 1st argument
        }
    }

    //----- formulate path-planning as QCQP

    let n = t_cap * 2;
    let m = t_cap - 2;
    let p = 6 * 2;

    let mut syms_p = vec![AMatBuild::new(MatType::SymPack(n)); m + 1];
    let vecs_q = vec![AMatBuild::new(MatType::General(n, 1)); m + 1];
    let mut scls_r = vec![0.; m + 1];

    let dt = 1. / t_cap as f64;
    let mut mat_d = AMatBuild::new(MatType::General(n, n));
    for i in 0 .. t_cap - 1 {
        let ti = i;
        mat_d[(ti, ti)] = -1. / dt;
        mat_d[(ti, ti + 1)] = 1. / dt;
        let ti = t_cap + i;
        mat_d[(ti, ti)] = -1. / dt;
        mat_d[(ti, ti + 1)] = 1. / dt;
    }
    // minimize total squared magnitude of velocity
    syms_p[0].set_by_fn(|r, c| {
        let mut v = 0.;
        for i in 0.. n {
            v += mat_d[(i, r)] * mat_d[(i, c)];
        }
        v
    });

    let dtdt = dt * dt;
    for i in 0 .. t_cap - 2 {
        let mut mat_d = AMatBuild::new(MatType::General(n, n));
        let mi = i + 1;
        let ti = i;
        mat_d[(ti, ti)] = 1. / dtdt;
        mat_d[(ti, ti + 1)] = -2. / dtdt;
        mat_d[(ti, ti + 2)] = 1. / dtdt;
        let ti = t_cap + i;
        mat_d[(ti, ti)] = 1. / dtdt;
        mat_d[(ti, ti + 1)] = -2. / dtdt;
        mat_d[(ti, ti + 2)] = 1. / dtdt;

        // limit acceleration magnitude
        syms_p[mi].set_by_fn(|r, c| {
            let mut v = 0.;
            for i in 0.. n {
                v += mat_d[(i, r)] * mat_d[(i, c)];
            }
            v
        });
        scls_r[mi] = -0.5 * a_cap * a_cap;
    }

    let mut mat_a = AMatBuild::new(MatType::General(p, n));
    let mut vec_b = AMatBuild::new(MatType::General(p, 1));

    // target point: x(0) = x_s, v(0) = 0
    // x0
    mat_a[(0, 0)] = 1.;
    vec_b[(0, 0)] = x_s.0;
    // x1
    mat_a[(1, t_cap)] = 1.;
    vec_b[(1, 0)] = x_s.0;
    // v0
    mat_a[(2, 0)] = -1.;
    mat_a[(2, 1)] = 1.;
    // v1
    mat_a[(3, t_cap)] = -1.;
    mat_a[(3, t_cap + 1)] = 1.;

    // target point: x(1) = x_t, v(1) = 0
    // x0
    mat_a[(4, t_cap - 1)] = 1.;
    vec_b[(4, 0)] = x_t.0;
    // x1
    mat_a[(5, t_cap * 2 - 1)] = 1.;
    vec_b[(5, 0)] = x_t.1;
    // v0
    mat_a[(6, t_cap - 2)] = -1.;
    mat_a[(6, t_cap - 1)] = 1.;
    // v1
    mat_a[(7, t_cap * 2 - 2)] = -1.;
    mat_a[(7, t_cap * 2 - 1)] = 1.;

    // target point: x(1/3) = x_m1
    let t_m1 = t_cap / 3;
    // x0
    mat_a[(8, t_m1)] = 1.;
    vec_b[(8, 0)] = x_m1.0;
    // x1
    mat_a[(9, t_cap + t_m1)] = 1.;
    vec_b[(9, 0)] = x_m1.1;

    // target point: x(2/3) = x_m2
    let t_m2 = t_cap * 2 / 3;
    // x0
    mat_a[(10, t_m2)] = 1.;
    vec_b[(10, 0)] = x_m2.0;
    // x1
    mat_a[(11, t_cap + t_m2)] = 1.;
    vec_b[(11, 0)] = x_m2.1;

    //----- solve QCQP

    let s = ASolver::new().par(|p| {
        p.eps_acc = 1e-3;
        utils::set_par_by_env(p);
    });
    let mut qp = AProbQCQP::new(syms_p, vecs_q, scls_r, mat_a, vec_b, s.par.eps_zero);
    let rslt = s.solve(qp.problem())?;
    //println!("{:?}", rslt);

    //----- graph plot 1

    let root = SVGBackend::new("plot1.svg", (480, 360)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(30)
        .x_label_area_size(35)
        .y_label_area_size(45)
        .build_cartesian_2d(
            -0.1..1.1,
            -1.7..1.7,
        )?;

    chart.configure_mesh()
        .x_labels(6)
        .y_labels(7)
        .x_desc("x0")
        .y_desc("x1")
        .disable_mesh()
        .draw()?;
    
    chart.draw_series(
        LineSeries::new(
            (0..t_cap).map(|i| (rslt.0[i], rslt.0[t_cap + i])),
            BLUE.stroke_width(2).filled()
        ).point_size(3)
    )?;

    chart.draw_series(
        PointSeries::of_element(
            [x_s, x_m1, x_m2, x_t],
            7,
            RED.mix(0.5).stroke_width(4),
            &|coord, size, style| {
                EmptyElement::at(coord)
                    + Cross::new((0, 0), size, style)
            }
        )
    )?;

    //----- graph plot 2

    let root = SVGBackend::new("plot2.svg", (480, 360)).into_drawing_area();
    root.fill(&WHITE)?;

    let (upper, lower) = root.split_vertically(360/2);

    let mut chart = ChartBuilder::on(&upper)
        .margin(20)
        .margin_bottom(0)
        .x_label_area_size(35)
        .y_label_area_size(45)
        .build_cartesian_2d(0.0..1.0, -1.0..12.0)?;

    chart.configure_mesh()
        .x_labels(6)
        .y_labels(5)
        .y_desc("velocity mag.")
        .disable_mesh()
        .draw()?;
    
    chart.draw_series(
        LineSeries::new(
            (0..t_cap - 1).map(|i| {
                let v0 = rslt.0[i + 1] - rslt.0[i];
                let v1 = rslt.0[t_cap + i + 1] - rslt.0[t_cap + i];
                let v = v0.hypot(v1) / dt;
                (i as f64 * dt, v)
            }),
            BLUE.stroke_width(1).filled()
        ).point_size(2)
    )?;
    
    let mut chart = ChartBuilder::on(&lower)
        .margin(20)
        .margin_top(0)
        .x_label_area_size(35)
        .y_label_area_size(45)
        .build_cartesian_2d(
             0.0..1.0,
            -5.0..95.0,
        )?;

    chart.configure_mesh()
        .x_labels(6)
        .y_labels(5)
        .x_desc("time")
        .y_desc("acceleration mag.")
        .disable_mesh()
        .draw()?;
    
    chart.draw_series(
        LineSeries::new(
            (0..t_cap - 2).map(|i| {
                let a0 = rslt.0[i + 2] - 2.0 * rslt.0[i + 1] + rslt.0[i];
                let a1 = rslt.0[t_cap + i + 2] - 2.0 * rslt.0[t_cap + i + 1] + rslt.0[t_cap + i];
                let a = a0.hypot(a1) / dt / dt;
                (i as f64 * dt, a)
            }),
            BLUE.stroke_width(1).filled()
        ).point_size(2)
    )?;

    Ok(())
}
