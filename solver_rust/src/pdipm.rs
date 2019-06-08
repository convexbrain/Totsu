/*!
Primal-dual interior point method
*/

use super::mat::{Mat, MatSlice, MatSliMu, FP, FP_MINPOS, FP_EPSILON};
use super::spmat::SpMat;
use super::matsvdsolve::SVDS;
use super::matlinalg::{LSQR, SpLinSolver};

const TOL_STEP: FP = FP_EPSILON;
const TOL_DIV0: FP = FP_MINPOS;

use std::io::Write;
macro_rules! writeln_or {
    ( $( $arg: expr ),* ) => {
        writeln!( $( $arg ),* ).or(Err(PDIPMErr::LogFailure))
    };
}

/**
A basic Primal-Dual Interior-Point Method solver struct.

<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML' async></script>

This struct abstracts a solver of continuous scalar convex optimization problem:
\\[
\\begin{array}{ll}
{\\rm minimize} & f_{\\rm obj}(x) \\\\
{\\rm subject \\ to} & f_i(x) \\le 0 \\quad (i = 0, \\ldots, m - 1) \\\\
& A x = b,
\\end{array}
\\]
where
- variables \\( x \\in {\\bf R}^n \\)
- \\( f_{\\rm obj}: {\\bf R}^n \\rightarrow {\\bf R} \\), convex and twice differentiable
- \\( f_i: {\\bf R}^n \\rightarrow {\\bf R} \\), convex and twice differentiable
- \\( A \\in {\\bf R}^{p \\times n} \\), \\( {\\bf rank} A = p < n \\), \\( b \\in {\\bf R}^p \\).

The solution gives optimal values of primal variables \\(x\\)
as well as dual variables \\(\\lambda \\in {\\bf R}^m\\) and \\(\\nu \\in {\\bf R}^p\\).
 */
pub struct PDIPM
{
    n_m_p: (usize, usize, usize),

    /***** matrix *****/
    // constant across loop
    a: Mat,
    b: Mat,
    // loop variable
    y: Mat,
    kkt: SpMat,
    // temporal in loop
    df_o: Mat,
    f_i: Mat,
    r_t: Mat,
    df_i: Mat,
    ddf: Mat,
}

/// Primal-Dual Interior-Point Method solver parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct PDIPMParam
{
    /// Tolerance of the surrogate duality gap.
    /// Tolerance of the primal and dual residuals.
    pub eps: FP,
    /// The factor to squeeze complementary slackness.
    pub mu: FP,
    /// The factor to decrease residuals in the backtracking line search.
    pub alpha: FP,
    /// The factor to decrease a step size in the backtracking line search.
    pub beta: FP,
    /// The factor to determine an initial step size in the backtracking line search.
    pub s_coef: FP,
    /// Initial margin value for dual variables of inequalities.
    pub margin: FP,
    /// Max iteration number of outer-loop for the Newton step.
    /// Max iteration number of inner-loop for the backtracking line search.
    pub n_loop: usize,
    /// Use iterative linear solver to calculate Newton step.
    pub use_iter: bool,
    /// Enables to log vector status.
    pub log_vecs: bool,
    /// Enables to log kkt matrix.
    pub log_kkt: bool
}

impl Default for PDIPMParam
{
    fn default() -> PDIPMParam
    {
        PDIPMParam {
            eps: 1e-8,
            mu: 10.,
            alpha: 0.1,
            beta: 0.8,
            s_coef: 0.99,
            margin: 1.,
            n_loop: 256,
            use_iter: false,
            log_vecs: false,
            log_kkt: false
        }
    }
}

/// Primal-Dual Interior-Point Method solver errors.
pub enum PDIPMErr<'a>
{
    /// Did not converged, and ended in an inaccurate result.
    NotConverged(&'a Mat),
    /// Did not meet inequality feasibility.
    Infeasible,
    /// Failed to log due to I/O error.
    LogFailure
}

impl<'a> From<PDIPMErr<'a>> for String
{
    fn from(err: PDIPMErr) -> String
    {
        match err {
            PDIPMErr::NotConverged(_) => "PDIPM Not Converged".into(),
            PDIPMErr::Infeasible => "PDIPM Infeasible".into(),
            PDIPMErr::LogFailure => "PDIPM Log Failure".into(),
        }
    }
}

impl PDIPM
{
    /// Creates an instance.
    pub fn new() -> PDIPM
    {
        PDIPM {
            n_m_p: (0, 0, 0),
            a: Mat::new(0, 0),
            b: Mat::new_vec(0),
            y: Mat::new_vec(0),
            kkt: SpMat::new(0, 0),
            df_o: Mat::new_vec(0),
            f_i: Mat::new_vec(0),
            r_t: Mat::new_vec(0),
            df_i: Mat::new(0, 0),
            ddf: Mat::new(0, 0)
        }
    }

    fn allocate(&mut self, n: usize, m: usize, p: usize)
    {
        if self.n_m_p != (n, m, p) {
            self.n_m_p = (n, m, p);
            self.a = Mat::new(p, n);
            self.b = Mat::new_vec(p);
            self.y = Mat::new_vec(n + m + p);
            self.kkt = SpMat::new(n + m + p, n + m + p);
            self.df_o = Mat::new_vec(n);
            self.f_i = Mat::new_vec(m);
            self.r_t = Mat::new_vec(n + m + p);
            self.df_i = Mat::new(m, n);
            self.ddf = Mat::new(n, n);
        }
    }

    /// Starts to solve a optimization problem by primal-dual interior-point method.
    /// 
    /// Returns `Ok` with optimal \\(x, \\lambda, \\nu\\) concatenated vector
    /// or `Err` with `PDIPMErr` type.
    /// * `param` is solver parameters.
    /// * `log` outputs solver progress.
    /// * `n` is \\(n\\), the dimension of the variable \\(x\\).
	/// * `m` is \\(m\\), the number of inequality constraints \\(f_i\\).
    /// * `p` is \\(p\\), the number of rows of equality constraints \\(A\\) and \\(b\\).
    /// * `d_objective(x, df_o)`
    ///   calculates first derivatives of the objective function
    ///   \\(\\nabla f_{\\rm obj}(x)\\).
    /// * `dd_objective(x, ddf_o)`
    ///   calculates second derivatives of the objective function
    ///   \\(\\nabla^2 f_{\\rm obj}(x)\\).
    /// * `inequality(x, f_i)`
    ///   calculates the inequality constraint functions
    ///   \\(f_i(x)\\).
    /// * `d_inequality(x, df_i)`
    ///   calculates first derivatives of the inequality constraint functions
    ///   \\(Df(x) = \\left[\\matrix{\\nabla f_0(x) & \\cdots & \\nabla f_{m-1}(x)}\\right]^T\\).
    /// * `dd_inequality(x, ddf_i, i)`
    ///   calculates second derivatives of the inequality constraint functions
    ///   \\(\\nabla^2 f_i(x)\\).
    /// * `equality(a, b)`
    ///   produces the equality constraints affine parameters \\(A\\) and \\(b\\).
    /// * `start_point(x)`
    ///   produces the initial values of \\(x\\).
    ///   **The initial values must satisfy all inequality constraints strictly: \\(f_i(x)<0\\).**
    ///   This may seem a hard requirement, but introducing **slack variables** helps in most cases.
    ///   Refer pre-defined solver implementations for example.
    pub fn solve<L, Fo1, Fo2, Fi0, Fi1, Fi2, Fe, Fs>(
        &mut self, param: &PDIPMParam, log: &mut L,
        n: usize, m: usize, p: usize,
        d_objective: Fo1,
        dd_objective: Fo2,
        inequality: Fi0,
        d_inequality: Fi1,
        dd_inequality: Fi2,
        equality: Fe,
        start_point: Fs
    ) -> Result<&Mat, PDIPMErr>
    where L: Write,
          Fo1: Fn(&MatSlice, &mut Mat),
          Fo2: Fn(&MatSlice, &mut Mat),
          Fi0: Fn(&MatSlice, &mut Mat),
          Fi1: Fn(&MatSlice, &mut Mat),
          Fi2: Fn(&MatSlice, &mut Mat, usize),
          Fe: FnOnce(&mut Mat, &mut Mat),
          Fs: FnOnce(MatSliMu)
    {
        let eps_feas = param.eps;
        let eps_eta = param.eps;
        let b_loop = param.n_loop;

        // allocate matrix
        self.allocate(n, m, p);

        // initialize
        let x = self.y.rows_mut(0 .. n);
        start_point(x);
        let mut lmd = self.y.rows_mut(n .. n + m);
        lmd.assign_all(param.margin);
        let mut nu = self.y.rows_mut(n + m .. n + m + p);
        nu.assign_all(0.);
        equality(&mut self.a, &mut self.b);

        // initial df_o, f_i, df_i
        let x = self.y.rows(0 .. n);
        d_objective(&x, &mut self.df_o);
        inequality(&x, &mut self.f_i);
        d_inequality(&x, &mut self.df_i);

        // inequality feasibility check
        if self.f_i.max().unwrap_or(-1.) >= 0. {return Err(PDIPMErr::Infeasible);}

        // initial residual - dual and primal
        let mut r_dual = self.r_t.rows_mut(0 .. n);
        r_dual.assign(&self.df_o);
        if m > 0 {
            let lmd = self.y.rows(n .. n + m);
            r_dual += self.df_i.t() * lmd;
        }
        if p > 0 {
            let nu = self.y.rows(n + m .. n + m + p);
            r_dual += self.a.t() * nu;
        }
        let mut r_pri = self.r_t.rows_mut(n + m .. n + m + p);
        if p > 0 {
            let x = self.y.rows(0 .. n);
            r_pri.assign(&(&self.a * x - &self.b));
        }

        //

        let mut cnt = 0;
        while cnt < param.n_loop {
            writeln_or!(log)?;
            writeln_or!(log, "===== ===== ===== ===== loop : {}", cnt)?;

            let x = self.y.rows(0 .. n);
            let lmd = self.y.rows(n .. n + m);

            /***** calc t *****/

            let eta = if m > 0 {-self.f_i.prod(&lmd)} else {eps_eta};

            // inequality feasibility check
            if eta < 0. {return Err(PDIPMErr::Infeasible);} // never happen

            let inv_t = if m > 0 {Some(eta / (param.mu * m as FP))} else {None};

            /***** update residual - central *****/

            if m > 0 {
                let mut r_cent = self.r_t.rows_mut(n .. n + m);
                r_cent.assign_s(&(lmd.diag_mul(&self.f_i) + inv_t.unwrap()), -1.);
            }

            /***** termination criteria *****/

            let r_dual = self.r_t.rows(0 .. n);
            let r_pri = self.r_t.rows(n + m .. n + m + p);

            let r_dual_norm = r_dual.norm_p2();
            let r_pri_norm = r_pri.norm_p2();

            writeln_or!(log, "|| r_dual || : {:.3e}", r_dual_norm)?;
            writeln_or!(log, "|| r_pri  || : {:.3e}", r_pri_norm)?;
            writeln_or!(log, "   eta       : {:.3e}", eta)?;

            if (r_dual_norm <= eps_feas) && (r_pri_norm <= eps_feas) && (eta <= eps_eta) {
                writeln_or!(log, "termination criteria satisfied")?;
                break;
            }

            /***** calc kkt matrix *****/
            
            let mut kkt_x_dual = self.kkt.slice_mut(0 .. n, 0 .. n);
            dd_objective(&x, &mut self.ddf);
            kkt_x_dual.assign(&self.ddf);
            for i in 0 .. m {
                dd_inequality(&x, &mut self.ddf, i);
                kkt_x_dual += lmd[(i, 0)] * &self.ddf;
            }

            if m > 0 {
                let mut kkt_lmd_dual = self.kkt.slice_mut(0 .. n, n .. n + m);
                kkt_lmd_dual.assign(&self.df_i.t());

                let mut kkt_x_cent = self.kkt.slice_mut(n .. n + m, 0 .. n);
                kkt_x_cent.assign_s(&lmd.diag_mul(&self.df_i), -1.);

                let mut kkt_lmd_cent = self.kkt.slice_mut(n .. n + m, n .. n + m);
                kkt_lmd_cent.assign_s(&self.f_i.clone_diag(), -1.);
            }

            if p > 0 {
                let mut kkt_nu_dual = self.kkt.slice_mut(0 .. n, n + m .. n + m + p);
                kkt_nu_dual.assign(&self.a.t());

                let mut kkt_x_pri = self.kkt.slice_mut(n + m .. n + m + p, 0 .. n);
                kkt_x_pri.assign(&self.a);
            }

            /***** calc search direction *****/

            if param.log_kkt {
                writeln_or!(log, "kkt : {}", self.kkt)?;
            }

            // negative dy
            let neg_dy = if param.use_iter {
                LSQR::new(self.kkt.size()).spsolve(&self.kkt, &self.r_t)
            }
            else {
                SVDS::new(self.kkt.size()).spsolve(&self.kkt, &self.r_t)
            };

            if param.log_vecs {
                writeln_or!(log, "y : {}", self.y.t())?;
                writeln_or!(log, "r_t : {}", self.r_t.t())?;
                writeln_or!(log, "neg_dy : {}", neg_dy.t())?;
            }

            /***** back tracking line search - from here *****/

            let mut s_max: FP = 1.;
            {
                let neg_dlmd = neg_dy.rows(n .. n + m);

                for i in 0 .. m {
                    if neg_dlmd[(i, 0)] > TOL_DIV0 { // to avoid zero-division by Dlmd
                        s_max = s_max.min(lmd[(i, 0)] / neg_dlmd[(i, 0)]);
                    }
                }
            }
            let mut s = param.s_coef * s_max;

            let mut y_p = &self.y - s * &neg_dy;

            let mut bcnt = 0;
            while bcnt < b_loop {
                let x_p = y_p.rows(0 .. n);
                let lmd_p = y_p.rows(n .. n + m);
                
                // update f_i
                inequality(&x_p, &mut self.f_i);

                if (self.f_i.max().unwrap_or(-1.) < 0.) && (lmd_p.min().unwrap_or(1.) > 0.) {break;}
                s *= param.beta;
                y_p = &self.y - s * &neg_dy;

                bcnt += 1;
            }

            writeln_or!(log, "s : {:.3e}", s)?;

            if bcnt < b_loop {
                writeln_or!(log, "feasible points found")?;
            }
            else {
                writeln_or!(log, "infeasible in this direction")?;
                return Err(PDIPMErr::NotConverged(&self.y));
            }

            let org_r_t_norm = self.r_t.norm_p2();

            while bcnt < b_loop {
                let x_p = y_p.rows(0 .. n);
                let lmd_p = y_p.rows(n .. n + m);
                let nu_p = y_p.rows(n + m .. n + m + p);

                // update df_o, f_i, df_i
                d_objective(&x_p, &mut self.df_o);
                inequality(&x_p, &mut self.f_i);
                d_inequality(&x_p, &mut self.df_i);

                // update residual
                let mut r_dual = self.r_t.rows_mut(0 .. n);
                r_dual.assign(&self.df_o);
                if m > 0 {
                    r_dual += self.df_i.t() * &lmd_p;
                }
                if p > 0 {
                    r_dual += self.a.t() * nu_p;
                }
                if m > 0 {
                    let mut r_cent = self.r_t.rows_mut(n .. n + m);
                    r_cent.assign_s(&(lmd_p.diag_mul(&self.f_i) + inv_t.unwrap()), -1.);
                }
                if p > 0 {
                    let mut r_pri = self.r_t.rows_mut(n + m .. n + m + p);
                    r_pri.assign(&(&self.a * x_p - &self.b));
                }

                if self.r_t.norm_p2() <= (1. - param.alpha * s) * org_r_t_norm {break;}
                s *= param.beta;
                y_p = &self.y - s * &neg_dy;

                bcnt += 1;
            }

            writeln_or!(log, "s : {:.3e}", s)?;

            if bcnt < b_loop {
                if (&y_p - &self.y).norm_p2() >= TOL_STEP {
                    writeln_or!(log, "update")?;
                    // update y
                    self.y.assign(&y_p);
                }
                else {
                    writeln_or!(log, "too small step")?;
                    return Err(PDIPMErr::NotConverged(&self.y));
                }
            }
            else {
                writeln_or!(log, "B-iteration limit")?;
                return Err(PDIPMErr::NotConverged(&self.y));
            }

            /***** back tracking line search - to here *****/

            cnt += 1;
        }

        if !(cnt < param.n_loop) {
            writeln_or!(log, "N-iteration limit")?;
            return Err(PDIPMErr::NotConverged(&self.y));
        }

        writeln_or!(log)?;
        writeln_or!(log, "===== ===== ===== ===== result")?;
        let x = self.y.rows(0 .. n);
        let lmd = self.y.rows(n .. n + m);
        let nu = self.y.rows(n + m .. n + m + p);
        writeln_or!(log, "x : {}", x.t())?;
        writeln_or!(log, "lmd : {}", lmd.t())?;
        writeln_or!(log, "nu : {}", nu.t())?;

        Ok(&self.y)
    }
}
