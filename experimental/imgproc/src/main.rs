use totsu::prelude::*;
use totsu::linalg::F64LAPACK;
use totsu::operator::MatBuild;

use utils;

use image::io::Reader as ImageReader;
use image::{Pixel, ImageBuffer, Luma};
use intel_mkl_src as _;
use anyhow::{Result, Context};

type LA = F64LAPACK;
type ASolver = Solver<LA, f64>;

//

struct Laplacian
{
    w: usize,
    h: usize,
}

impl Laplacian
{
    fn new(w: usize, h: usize) -> Self
    {
        Laplacian {
            w,
            h,
        }
    }
}

impl Operator<f64> for Laplacian
{
    fn size(&self) -> (usize, usize)
    {
        ((self.w - 2) * (self.h - 2), self.w * self.h)
    }

    fn op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        LA::scale(beta, y);

        let mut y_rest = y;
        for cy in 0..(self.h - 2) {
            let (y_line, y_lh) = y_rest.split_at_mut(self.w - 2);
            y_rest = y_lh;

            let x_0 = x.split_at(cy * self.w).1;
            let x_00 = x_0.split_at(self.w - 2).0;
            let x_01 = x_0.split_at(1).1.split_at(self.w - 2).0;
            let x_02 = x_0.split_at(2).1.split_at(self.w - 2).0;
            LA::add(alpha * -0.125, x_00, y_line);
            LA::add(alpha * -0.125, x_01, y_line);
            LA::add(alpha * -0.125, x_02, y_line);

            let x_1 = x.split_at((cy + 1) * self.w).1;
            let x_10 = x_1.split_at(self.w - 2).0;
            let x_11 = x_1.split_at(1).1.split_at(self.w - 2).0;
            let x_12 = x_1.split_at(2).1.split_at(self.w - 2).0;
            LA::add(alpha * -0.125, x_10, y_line);
            LA::add(alpha *  1.0  , x_11, y_line);
            LA::add(alpha * -0.125, x_12, y_line);

            let x_2 = x.split_at((cy + 2) * self.w).1;
            let x_20 = x_2.split_at(self.w - 2).0;
            let x_21 = x_2.split_at(1).1.split_at(self.w - 2).0;
            let x_22 = x_2.split_at(2).1.split_at(self.w - 2).0;
            LA::add(alpha * -0.125, x_20, y_line);
            LA::add(alpha * -0.125, x_21, y_line);
            LA::add(alpha * -0.125, x_22, y_line);
        }
    }

    fn trans_op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        LA::scale(beta, y);
        
        let mut x_rest = x;
        for cy in 0..(self.h - 2) {
            let (x_line, x_lh) = x_rest.split_at(self.w - 2);
            x_rest = x_lh;

            let y_0 = y.split_at_mut(cy * self.w).1;
            let y_00 = y_0.split_at_mut(self.w - 2).0;
            LA::add(alpha * -0.125, x_line, y_00);
            let y_01 = y_0.split_at_mut(1).1.split_at_mut(self.w - 2).0;
            LA::add(alpha * -0.125, x_line, y_01);
            let y_02 = y_0.split_at_mut(2).1.split_at_mut(self.w - 2).0;
            LA::add(alpha * -0.125, x_line, y_02);

            let y_1 = y.split_at_mut((cy + 1) * self.w).1;
            let y_10 = y_1.split_at_mut(self.w - 2).0;
            LA::add(alpha * -0.125, x_line, y_10);
            let y_11 = y_1.split_at_mut(1).1.split_at_mut(self.w - 2).0;
            LA::add(alpha *  1.0  , x_line, y_11);
            let y_12 = y_1.split_at_mut(2).1.split_at_mut(self.w - 2).0;
            LA::add(alpha * -0.125, x_line, y_12);

            let y_2 = y.split_at_mut((cy + 2) * self.w).1;
            let y_20 = y_2.split_at_mut(self.w - 2).0;
            LA::add(alpha * -0.125, x_line, y_20);
            let y_21 = y_2.split_at_mut(1).1.split_at_mut(self.w - 2).0;
            LA::add(alpha * -0.125, x_line, y_21);
            let y_22 = y_2.split_at_mut(2).1.split_at_mut(self.w - 2).0;
            LA::add(alpha * -0.125, x_line, y_22);
        }
    }
}

//

struct ProbOpC
{
    x_sz: usize,
    t_sz: usize,
}

impl ProbOpC
{
    fn new(width: usize, height: usize) -> Self
    {
        ProbOpC {
            x_sz: width * height,
            t_sz: (width - 2) * (height - 2),
        }
    }
}

impl Operator<f64> for ProbOpC
{
    fn size(&self) -> (usize, usize)
    {
        (self.x_sz + 1 + self.t_sz, 1)
    }

    fn op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        LA::scale(beta, y);
        y[self.x_sz] += alpha * x[0];
    }

    fn trans_op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        y[0] = alpha * x[self.x_sz] + beta * y[0];
    }
}

//

struct ProbOpA
{
    x_sz: usize,
    t_sz: usize,
    lap: Laplacian,
    one: MatBuild<LA, f64>,
}

impl ProbOpA
{
    fn new(width: usize, height: usize) -> Self
    {
        ProbOpA {
            x_sz: width * height,
            t_sz: (width - 2) * (height - 2),
            lap: Laplacian::new(width, height),
            one: MatBuild::new(MatType::General((width - 2) * (height - 2), 1))
                 .by_fn(|_, _| 1.0),
        }
    }
}

impl Operator<f64> for ProbOpA
{
    fn size(&self) -> (usize, usize)
    {
        (self.t_sz * 2 + 1 + self.x_sz * 2 + 1 + self.x_sz, self.x_sz + 1 + self.t_sz)
    }

    fn op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        let (x_x, x_rest) = x.split_at(self.x_sz);
        let (x_z, x_t) = x_rest.split_at(1);

        let (y_lp, y_rest) = y.split_at_mut(self.t_sz);
        let (y_ln, y_rest) = y_rest.split_at_mut(self.t_sz);
        let (y_l1, y_rest) = y_rest.split_at_mut(1);
        let (y_xp, y_rest) = y_rest.split_at_mut(self.x_sz);
        let (y_xn, y_rest) = y_rest.split_at_mut(self.x_sz);
        let (y_sz, y_sx) = y_rest.split_at_mut(1);

        self.lap.op(alpha, x_x, beta, y_lp);
        LA::add(-alpha, x_t, y_lp);

        self.lap.op(-alpha, x_x, beta, y_ln);
        LA::add(-alpha, x_t, y_ln);

        self.one.trans_op(alpha, x_t, beta, y_l1);

        LA::scale(beta, y_xp);
        LA::add(-alpha, x_x, y_xp);

        LA::scale(beta, y_xn);
        LA::add(alpha, x_x, y_xn);

        y_sz[0] = -alpha * x_z[0] + beta * y_sz[0];

        LA::scale(beta, y_sx);
        LA::add(-alpha, x_x, y_sx);
    }

    fn trans_op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        let (y_x, y_rest) = y.split_at_mut(self.x_sz);
        let (y_z, y_t) = y_rest.split_at_mut(1);

        let (x_lp, x_rest) = x.split_at(self.t_sz);
        let (x_ln, x_rest) = x_rest.split_at(self.t_sz);
        let (x_l1, x_rest) = x_rest.split_at(1);
        let (x_xp, x_rest) = x_rest.split_at(self.x_sz);
        let (x_xn, x_rest) = x_rest.split_at(self.x_sz);
        let (x_sz, x_sx) = x_rest.split_at(1);

        self.lap.trans_op(alpha, x_lp, beta, y_x);
        self.lap.trans_op(-alpha, x_ln, 1.0, y_x);
        LA::add(-alpha, x_xp, y_x);
        LA::add(alpha, x_xn, y_x);
        LA::add(-alpha, x_sx, y_x);

        y_z[0] = -alpha * x_sz[0] + beta * y_z[0];

        self.one.op(alpha, x_l1, beta, y_t);
        LA::add(-alpha, x_lp, y_t);
        LA::add(-alpha, x_ln, y_t);
    }
}

//

struct ProbOpB<'a>
{
    x_sz: usize,
    t_sz: usize,
    lambda_lxh_norm1: f64,
    one: MatBuild<LA, f64>,
    xh: MatOp<'a, LA, f64>,
}

impl<'a> ProbOpB<'a>
{
    fn new(width: usize, height: usize, lambda: f64, vec_xh: &'a[f64]) -> Self
    {
        let mut lxh = vec![0.0; (width - 2) * (height - 2)];
        Laplacian::new(width, height).op(1.0, vec_xh, 0.0, &mut lxh);
        let lambda_lxh_norm1 = lambda * LA::abssum(&lxh);

        ProbOpB {
            x_sz: width * height,
            t_sz: (width - 2) * (height - 2),
            lambda_lxh_norm1,
            one: MatBuild::new(MatType::General(width * height, 1))
                 .by_fn(|_, _| 1.0),
            xh: MatOp::new(MatType::General(width * height, 1), vec_xh),
        }
    }
}

impl<'a> Operator<f64> for ProbOpB<'a>
{
    fn size(&self) -> (usize, usize)
    {
        (self.t_sz * 2 + 1 + self.x_sz * 2 + 1 + self.x_sz, 1)
    }

    fn op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        let (y_lp_ln, y_rest) = y.split_at_mut(self.t_sz * 2);
        let (y_l1, y_rest) = y_rest.split_at_mut(1);
        let (y_xp, y_rest) = y_rest.split_at_mut(self.x_sz);
        let (y_xn, y_rest) = y_rest.split_at_mut(self.x_sz);
        let (y_sz, y_sx) = y_rest.split_at_mut(1);

        LA::scale(beta, y_lp_ln);

        y_l1[0] = alpha * self.lambda_lxh_norm1 * x[0] + beta * y_l1[0];

        LA::scale(beta, y_xp);

        self.one.op(alpha, x, beta, y_xn);

        LA::scale(beta, y_sz);

        self.xh.op(-alpha, x, beta, y_sx);
    }

    fn trans_op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        let (_x_lp_ln, x_rest) = x.split_at(self.t_sz * 2);
        let (x_l1, x_rest) = x_rest.split_at(1);
        let (_x_xp, x_rest) = x_rest.split_at(self.x_sz);
        let (x_xn, x_rest) = x_rest.split_at(self.x_sz);
        let (_x_sz, x_sx) = x_rest.split_at(1);

        self.one.trans_op(alpha, x_xn, beta, y);
        self.xh.trans_op(-alpha, x_sx, 1.0, y);
        y[0] += alpha * self.lambda_lxh_norm1 * x_l1[0];
    }
}

//

struct ProbCone
{
    x_sz: usize,
    t_sz: usize,
    rpos: ConeRPos<f64>,
    soc: ConeSOC<LA, f64>,
}

impl ProbCone
{
    fn new(width: usize, height: usize) -> Self
    {
        ProbCone {
            x_sz: width * height,
            t_sz: (width - 2) * (height - 2),
            rpos: ConeRPos::new(),
            soc: ConeSOC::new(),
        }
    }
}

impl Cone<f64> for ProbCone
{
    fn proj(&mut self, dual_cone: bool, x: &mut[f64]) -> Result<(), ()>
    {
        let (x_rpos, x_soc) = x.split_at_mut(self.t_sz * 2 + 1 + self.x_sz * 2);

        self.rpos.proj(dual_cone, x_rpos)?;
        self.soc.proj(dual_cone, x_soc)?;

        Ok(())
    }

    fn product_group<G: Fn(&mut[f64]) + Copy>(&self, dp_tau: &mut[f64], group: G)
    {
        let (dp_tau_rpos, dp_tau_soc) = dp_tau.split_at_mut(self.t_sz * 2 + 1 + self.x_sz * 2);

        self.rpos.product_group(dp_tau_rpos, group);
        self.soc.product_group(dp_tau_soc, group);
    }
}

//

fn main() -> Result<()> {
    env_logger::init();

    //----- read input image

    let in_img_file = std::env::args().nth(1).context("no input image filename")?;
    println!("{}", in_img_file);
    let in_img = ImageReader::open(in_img_file)?.decode()?;

    let in_mono = in_img.to_luma8();

    let width = in_mono.width();
    let height = in_mono.height();
    let vec_xh: Vec<f64> = in_mono.pixels()
                           .flat_map(|p| p.channels())
                           .map(|p| {*p as f64 / 255.0})
                           .collect();
    println!("{}x{}={}", width, height, vec_xh.len());

    //-----

    let lambda = utils::num_by_env("LAMBDA").unwrap_or(0.5);

    //----- solve user-defined problem

    let s = ASolver::new().par(|p| {
        p.eps_acc = 1e-3;
        utils::set_par_by_env(p);
    });
    let op_c = ProbOpC::new(width as usize, height as usize);
    let op_a = ProbOpA::new(width as usize, height as usize);
    let op_b = ProbOpB::new(width as usize, height as usize, lambda, &vec_xh);
    let cone = ProbCone::new(width as usize, height as usize);
    let mut work = vec![0.0; ASolver::query_worklen(op_a.size())];
    let rslt = s.solve((op_c, op_a, op_b, cone, &mut work))?;
    //println!("{:?}", rslt);

    //----- write output image

    let vec_x = rslt.0.split_at((width * height) as usize).0;
    let out_mono = ImageBuffer::from_fn(width, height, |x, y| {
        let v = vec_x[(x + y * width) as usize];
        let v = if v < 0. {0.} else {v};
        let v = if v > 1. {1.} else {v};
        let v = (v * 255.) as u8;
        Luma::from([v])
    });
    out_mono.save("out.png")?;

    Ok(())
}
