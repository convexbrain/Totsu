/*!
Primal-dual interior point method
*/

use super::mat::{Mat, MatSlice, MatSliMu, FP, FP_MINPOS, FP_EPSILON};
use super::matsvd::MatSVD;

use std::io::Write;
macro_rules! writeln_or {
    ( $( $arg: expr ),* ) => {
        writeln!( $( $arg ),* ).or(Err("log: I/O Error"))
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
    mat_a: Mat,
    vec_b: Mat,
    // loop variable
    vec_y: Mat,
    kkt: Mat,
    // temporal in loop
    vec_df_o: Mat,
    vec_f_i: Mat,
    vec_r_t: Mat,
    mat_df_i: Mat,
    mat_ddf: Mat,

    /***** KKT matrix decomposition solver *****/
    svd: MatSVD
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
    /// Enables to warm-start svd.
    pub svd_warm: bool,
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
            svd_warm: true,
            log_kkt: false
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
            mat_a: Mat::new(0, 0),
            vec_b: Mat::new_vec(0),
            vec_y: Mat::new_vec(0),
            kkt: Mat::new(0, 0),
            vec_df_o: Mat::new_vec(0),
            vec_f_i: Mat::new_vec(0),
            vec_r_t: Mat::new_vec(0),
            mat_df_i: Mat::new(0, 0),
            mat_ddf: Mat::new(0, 0),
            svd: MatSVD::new((0, 0))
        }
    }

    fn allocate(&mut self, n: usize, m: usize, p: usize)
    {
        if self.n_m_p != (n, m, p) {
            self.n_m_p = (n, m, p);
            self.mat_a = Mat::new(p, n);
            self.vec_b = Mat::new_vec(p);
            self.vec_y = Mat::new_vec(n + m + p);
            self.kkt = Mat::new(n + m + p, n + m + p);
            self.vec_df_o = Mat::new_vec(n);
            self.vec_f_i = Mat::new_vec(m);
            self.vec_r_t = Mat::new_vec(n + m + p);
            self.mat_df_i = Mat::new(m, n);
            self.mat_ddf = Mat::new(n, n);
            self.svd = MatSVD::new((n + m + p, n + m + p));
        }
    }

    /// Starts to solve a optimization problem by primal-dual interior-point method.
    /// 
    /// Returns `Ok` with optimal \\(x, \\lambda, \\nu\\) concatenated vector
    /// or `Err` with message string.
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
    ) -> Result<&Mat, &'static str>
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
        let b_loop = param.n_loop;

        // parameter check
        if n == 0 {return Err("n: 0");}

        // allocate matrix
        self.allocate(n, m, p);

        // initialize
        let x = self.vec_y.rows_mut(0 .. n);
        start_point(x);
        let mut lmd = self.vec_y.rows_mut(n .. n + m);
        lmd.assign_all(param.margin);
        equality(&mut self.mat_a, &mut self.vec_b);

        // initial df_o, f_i, df_i
        let x = self.vec_y.rows(0 .. n);
        d_objective(&x, &mut self.vec_df_o);
        inequality(&x, &mut self.vec_f_i);
        d_inequality(&x, &mut self.mat_df_i);

        // inequality feasibility check
        if self.vec_f_i.max().unwrap_or(-1.) >= 0. {return Err("inequality: not feasible at init");}

        // initial residual - dual and primal
        let mut r_dual = self.vec_r_t.rows_mut(0 .. n);
        r_dual.assign(&self.vec_df_o);
        if m > 0 {
            let lmd = self.vec_y.rows(n .. n + m);
            r_dual += self.mat_df_i.t() * lmd;
        }
        if p > 0 {
            let nu = self.vec_y.rows(n + m .. n + m + p);
            r_dual += self.mat_a.t() * nu;
        }
        let mut r_pri = self.vec_r_t.rows_mut(n + m .. n + m + p);
        if p > 0 {
            let x = self.vec_y.rows(0 .. n);
            r_pri.assign(&(&self.mat_a * x - &self.vec_b));
        }

        //

        let mut cnt = 0;
        while cnt < param.n_loop {
            writeln_or!(log)?;
            writeln_or!(log, "===== ===== ===== ===== loop : {}", cnt)?;

            let x = self.vec_y.rows(0 .. n);
            let lmd = self.vec_y.rows(n .. n + m);

            /***** calc t *****/

            let eta = if m > 0 {
                -self.vec_f_i.prod(&lmd)
            }
            else {
                param.eps
            };

            // inequality feasibility check
            if eta < 0. {return Err("inequality: not feasible in loop");}

            let inv_t = eta / (param.mu * m as FP);

            /***** update residual - central *****/

            if m > 0 {
                let mut r_cent = self.vec_r_t.rows_mut(n .. n + m);
                r_cent.assign(&(-lmd.clone_diag() * &self.vec_f_i - inv_t));
            }

            /***** termination criteria *****/

            let r_dual = self.vec_r_t.rows(0 .. n);
            let r_pri = self.vec_r_t.rows(n + m .. n + m + p);

            let r_dual_norm = r_dual.norm_p2();
            let r_pri_norm = r_pri.norm_p2();

            writeln_or!(log, "|| r_dual || : {:.3e}", r_dual_norm)?;
            writeln_or!(log, "|| r_pri  || : {:.3e}", r_pri_norm)?;
            writeln_or!(log, "   eta       : {:.3e}", eta)?;

            if (r_dual_norm <= eps_feas) && (r_pri_norm <= eps_feas) && (eta <= param.eps) {
                writeln_or!(log, "termination criteria satisfied")?;
                break;
            }

            /***** calc kkt matrix *****/
            
            let mut kkt_x_dual = self.kkt.slice_mut(0 .. n, 0 .. n);
            dd_objective(&x, &mut self.mat_ddf);
            kkt_x_dual.assign(&self.mat_ddf);
            for i in 0 .. m {
                dd_inequality(&x, &mut self.mat_ddf, i);
                kkt_x_dual += lmd[(i, 0)] * &self.mat_ddf;
            }

            if m > 0 {
                let mut kkt_lmd_dual = self.kkt.slice_mut(0 .. n, n .. n + m);
                kkt_lmd_dual.assign(&self.mat_df_i.t());

                let mut kkt_x_cent = self.kkt.slice_mut(n .. n + m, 0 .. n);
                kkt_x_cent.assign(&(-lmd.clone_diag() * &self.mat_df_i));

                let mut kkt_lmd_cent = self.kkt.slice_mut(n .. n + m, n .. n + m);
                kkt_lmd_cent.assign(&(-self.vec_f_i.clone_diag()));
            }

            if p > 0 {
                let mut kkt_nu_dual = self.kkt.slice_mut(0 .. n, n + m .. n + m + p);
                kkt_nu_dual.assign(&self.mat_a.t());

                let mut kkt_x_pri = self.kkt.slice_mut(n + m .. n + m + p, 0 .. n);
                kkt_x_pri.assign(&self.mat_a);
            }

            /***** calc search direction *****/

            if param.log_kkt {
                writeln_or!(log, "kkt : {}", self.kkt)?;
            }

            if param.svd_warm {
                self.svd.decomp_warm(&self.kkt);
            }
            else {
                self.svd.decomp(&self.kkt);
            }
            
            let dy = self.svd.solve(&(-&self.vec_r_t));

            writeln_or!(log, "y : {}", self.vec_y.t())?;
            writeln_or!(log, "r_t : {}", self.vec_r_t.t())?;
            writeln_or!(log, "dy : {}", dy.t())?;

            /***** back tracking line search - from here *****/

            let mut s_max: FP = 1.;
            {
                let dlmd = dy.rows(n .. n + m);

                for i in 0 .. m {
                    if dlmd[(i, 0)] < -FP_MINPOS { // to avoid zero-division by Dlmd
                        s_max = s_max.min(-lmd[(i, 0)] / dlmd[(i, 0)]);
                    }
                }
            }
            let mut s = param.s_coef * s_max;

            let mut y_p = &self.vec_y + s * &dy;

            let mut bcnt = 0;
            while bcnt < b_loop {
                let x_p = y_p.rows(0 .. n);
                let lmd_p = y_p.rows(n .. n + m);
                
                // update f_i
                inequality(&x_p, &mut self.vec_f_i);

                if (self.vec_f_i.max().unwrap_or(-1.) < 0.) && (lmd_p.min().unwrap_or(1.) > 0.) {break;}
                s *= param.beta;
                y_p = &self.vec_y + s * &dy;

                bcnt += 1;
            }

            writeln_or!(log, "s : {:.3e}", s)?;

            if bcnt < b_loop {
                writeln_or!(log, "feasible points found")?;
            }
            else {
                writeln_or!(log, "infeasible in this direction")?;
            }

            let org_r_t_norm = self.vec_r_t.norm_p2();

            while bcnt < b_loop {
                let x_p = y_p.rows(0 .. n);
                let lmd_p = y_p.rows(n .. n + m);
                let nu_p = y_p.rows(n + m .. n + m + p);

                // update df_o, f_i, df_i
                d_objective(&x_p, &mut self.vec_df_o);
                inequality(&x_p, &mut self.vec_f_i);
                d_inequality(&x_p, &mut self.mat_df_i);

                // update residual
                let mut r_dual = self.vec_r_t.rows_mut(0 .. n);
                r_dual.assign(&self.vec_df_o);
                if m > 0 {
                    r_dual += self.mat_df_i.t() * &lmd_p;
                }
                if p > 0 {
                    r_dual += self.mat_a.t() * nu_p;
                }
                if m > 0 {
                    let mut r_cent = self.vec_r_t.rows_mut(n .. n + m);
                    r_cent.assign(&(-lmd_p.clone_diag() * &self.vec_f_i - inv_t));
                }
                if p > 0 {
                    let mut r_pri = self.vec_r_t.rows_mut(n + m .. n + m + p);
                    r_pri.assign(&(&self.mat_a * x_p - &self.vec_b));
                }

                if self.vec_r_t.norm_p2() <= (1. - param.alpha * s) * org_r_t_norm {break;}
                s *= param.beta;
                y_p = &self.vec_y + s * &dy;

                bcnt += 1;
            }

            writeln_or!(log, "s : {:.3e}", s)?;

            if (bcnt < b_loop) && ((&y_p - &self.vec_y).norm_p2() >= FP_EPSILON) {
                writeln_or!(log, "update")?;
                // update y
                self.vec_y.assign(&y_p);
            }
            else {
                writeln_or!(log, "no more improvement")?;
                return Err("line search: not converged");
            }

            /***** back tracking line search - to here *****/

            cnt += 1;
        }

        if !(cnt < param.n_loop) {
            writeln_or!(log, "iteration limit")?;
            return Err("iteration: not converged");
        }

        writeln_or!(log)?;
        writeln_or!(log, "===== ===== ===== ===== result")?;
        let x = self.vec_y.rows(0 .. n);
        let lmd = self.vec_y.rows(n .. n + m);
        let nu = self.vec_y.rows(n + m .. n + m + p);
        writeln_or!(log, "x : {}", x.t())?;
        writeln_or!(log, "lmd : {}", lmd.t())?;
        writeln_or!(log, "nu : {}", nu.t())?;

        Ok(&self.vec_y)
    }
}
