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

impl PDIPM
{
    /// Creates an instance with default parameters.
    pub fn new() -> PDIPM
    {
        PDIPM {
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

    /// Starts to solve a optimization problem by primal-dual interior-point method.
    /// 
    /// Returns `Ok` with optimal \\(x, \\lambda, \\nu\\) concatenated vector
    /// or `Err` with message string.
    /// * `n` is \\(n\\), the dimension of the variable \\(x\\).
	/// * `m` is \\(m\\), the number of inequality constraints \\(f_i\\).
    /// * `p` is \\(p\\), the number of rows of equality constraints \\(A\\) and \\(b\\).
    /// * `log` outputs solver progress.
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
    pub fn solve<L, Fo1, Fo2, Fi0, Fi1, Fi2, Fe, Fs>(&self,
        n: usize, m: usize, p: usize,
        log: &mut L,
        d_objective: Fo1,
        dd_objective: Fo2,
        inequality: Fi0,
        d_inequality: Fi1,
        dd_inequality: Fi2,
        equality: Fe,
        start_point: Fs
    ) -> Result<Mat, &'static str>
    where L: Write,
          Fo1: Fn(&MatSlice, &mut Mat),
          Fo2: Fn(&MatSlice, &mut Mat),
          Fi0: Fn(&MatSlice, &mut Mat),
          Fi1: Fn(&MatSlice, &mut Mat),
          Fi2: Fn(&MatSlice, &mut Mat, usize),
          Fe: FnOnce(&mut Mat, &mut Mat),
          Fs: FnOnce(MatSliMu)
    {
        let eps_feas = self.eps;
        let b_loop = self.n_loop;

        // parameter check
        if n == 0 {return Err("n: 0");}

        /***** matrix *****/
        // constant across loop
        let mut mat_a = Mat::new(p, n);
        let mut vec_b = Mat::new_vec(p);
        // loop variable
        let mut vec_y = Mat::new_vec(n + m + p);
        let mut kkt = Mat::new(n + m + p, n + m + p);
        // temporal in loop
        let mut vec_df_o = Mat::new_vec(n);
        let mut vec_f_i = Mat::new_vec(m);
        let mut vec_r_t = Mat::new_vec(n + m + p);
        let mut mat_df_i = Mat::new(m, n);
        let mut mat_ddf = Mat::new(n, n);

        /***** KKT matrix decomposition solver *****/
        let mut svd = MatSVD::new(kkt.size());
        
        // initialize
        let x = vec_y.rows_mut(0 .. n);
        start_point(x);
        let mut lmd = vec_y.rows_mut(n .. n + m);
        lmd.assign_all(self.margin);
        equality(&mut mat_a, &mut vec_b);

        // initial df_o, f_i, df_i
        let x = vec_y.rows(0 .. n);
        d_objective(&x, &mut vec_df_o);
        inequality(&x, &mut vec_f_i);
        d_inequality(&x, &mut mat_df_i);

        // inequality feasibility check
        if vec_f_i.max().unwrap_or(-1.) >= 0. {return Err("inequality: not feasible at init");}

        // initial residual - dual and primal
        let mut r_dual = vec_r_t.rows_mut(0 .. n);
        r_dual.assign(&vec_df_o);
        if m > 0 {
            let lmd = vec_y.rows(n .. n + m);
            r_dual += mat_df_i.t() * lmd;
        }
        if p > 0 {
            let nu = vec_y.rows(n + m .. n + m + p);
            r_dual += mat_a.t() * nu;
        }
        let mut r_pri = vec_r_t.rows_mut(n + m .. n + m + p);
        if p > 0 {
            let x = vec_y.rows(0 .. n);
            r_pri.assign(&(&mat_a * x - &vec_b));
        }

        //

        let mut cnt = 0;
        while cnt < self.n_loop {
            writeln_or!(log)?;
            writeln_or!(log, "===== ===== ===== ===== loop : {}", cnt)?;

            let x = vec_y.rows(0 .. n);
            let lmd = vec_y.rows(n .. n + m);

            /***** calc t *****/

            let eta = if m > 0 {
                -(vec_f_i.t() * &lmd)[(0, 0)]
            }
            else {
                self.eps
            };

            // inequality feasibility check
            if eta < 0. {return Err("inequality: not feasible in loop");}

            let inv_t = eta / (self.mu * m as FP);

            /***** update residual - central *****/

            if m > 0 {
                let mut r_cent = vec_r_t.rows_mut(n .. n + m);
                r_cent.assign(&(-lmd.clone_diag() * &vec_f_i - inv_t));
            }

            /***** termination criteria *****/

            let r_dual = vec_r_t.rows(0 .. n);
            let r_pri = vec_r_t.rows(n + m .. n + m + p);

            let r_dual_norm = r_dual.norm_p2();
            let r_pri_norm = r_pri.norm_p2();

            writeln_or!(log, "|| r_dual || : {:.3e}", r_dual_norm)?;
            writeln_or!(log, "|| r_pri  || : {:.3e}", r_pri_norm)?;
            writeln_or!(log, "   eta       : {:.3e}", eta)?;

            if (r_dual_norm <= eps_feas) && (r_pri_norm <= eps_feas) && (eta <= self.eps) {
                writeln_or!(log, "termination criteria satisfied")?;
                break;
            }

            /***** calc kkt matrix *****/
            
            let mut kkt_x_dual = kkt.slice_mut(0 .. n, 0 .. n);
            dd_objective(&x, &mut mat_ddf);
            kkt_x_dual.assign(&mat_ddf);
            for i in 0 .. m {
                dd_inequality(&x, &mut mat_ddf, i);
                kkt_x_dual += lmd[(i, 0)] * &mat_ddf;
            }

            if m > 0 {
                let mut kkt_lmd_dual = kkt.slice_mut(0 .. n, n .. n + m);
                kkt_lmd_dual.assign(&mat_df_i.t());

                let mut kkt_x_cent = kkt.slice_mut(n .. n + m, 0 .. n);
                kkt_x_cent.assign(&(-lmd.clone_diag() * &mat_df_i));

                let mut kkt_lmd_cent = kkt.slice_mut(n .. n + m, n .. n + m);
                kkt_lmd_cent.assign(&(-vec_f_i.clone_diag()));
            }

            if p > 0 {
                let mut kkt_nu_dual = kkt.slice_mut(0 .. n, n + m .. n + m + p);
                kkt_nu_dual.assign(&mat_a.t());

                let mut kkt_x_pri = kkt.slice_mut(n + m .. n + m + p, 0 .. n);
                kkt_x_pri.assign(&mat_a);
            }

            /***** calc search direction *****/

            if self.log_kkt {
                writeln_or!(log, "kkt : {}", kkt)?;
            }

            if self.svd_warm {
                svd.decomp_warm(&kkt);
            }
            else {
                svd.decomp(&kkt);
            }
            
            let dy = svd.solve(&(-&vec_r_t));

            writeln_or!(log, "y : {}", vec_y.t())?;
            writeln_or!(log, "r_t : {}", vec_r_t.t())?;
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
            let mut s = self.s_coef * s_max;

            let mut y_p = &vec_y + s * &dy;

            let mut bcnt = 0;
            while bcnt < b_loop {
                let x_p = y_p.rows(0 .. n);
                let lmd_p = y_p.rows(n .. n + m);
                
                // update f_i
                inequality(&x_p, &mut vec_f_i);

                if (vec_f_i.max().unwrap_or(-1.) < 0.) && (lmd_p.min().unwrap_or(1.) > 0.) {break;}
                s *= self.beta;
                y_p = &vec_y + s * &dy;

                bcnt += 1;
            }

            writeln_or!(log, "s : {:.3e}", s)?;

            if bcnt < b_loop {
                writeln_or!(log, "feasible points found")?;
            }
            else {
                writeln_or!(log, "infeasible in this direction")?;
            }

            let org_r_t_norm = vec_r_t.norm_p2();

            while bcnt < b_loop {
                let x_p = y_p.rows(0 .. n);
                let lmd_p = y_p.rows(n .. n + m);
                let nu_p = y_p.rows(n + m .. n + m + p);

                // update df_o, f_i, df_i
                d_objective(&x_p, &mut vec_df_o);
                inequality(&x_p, &mut vec_f_i);
                d_inequality(&x_p, &mut mat_df_i);

                // update residual
                let mut r_dual = vec_r_t.rows_mut(0 .. n);
                r_dual.assign(&vec_df_o);
                if m > 0 {
                    r_dual += mat_df_i.t() * &lmd_p;
                }
                if p > 0 {
                    r_dual += mat_a.t() * nu_p;
                }
                if m > 0 {
                    let mut r_cent = vec_r_t.rows_mut(n .. n + m);
                    r_cent.assign(&(-lmd_p.clone_diag() * &vec_f_i - inv_t));
                }
                if p > 0 {
                    let mut r_pri = vec_r_t.rows_mut(n + m .. n + m + p);
                    r_pri.assign(&(&mat_a * x_p - &vec_b));
                }

                if vec_r_t.norm_p2() <= (1. - self.alpha * s) * org_r_t_norm {break;}
                s *= self.beta;
                y_p = &vec_y + s * &dy;

                bcnt += 1;
            }

            writeln_or!(log, "s : {:.3e}", s)?;

            if (bcnt < b_loop) && ((&y_p - &vec_y).norm_p2() >= FP_EPSILON) {
                writeln_or!(log, "update")?;
                // update y
                vec_y.assign(&y_p);
            }
            else {
                writeln_or!(log, "no more improvement")?;
                return Err("line search: not converged");
            }

            /***** back tracking line search - to here *****/

            cnt += 1;
        }

        if !(cnt < self.n_loop) {
            writeln_or!(log, "iteration limit")?;
            return Err("iteration: not converged");
        }

        writeln_or!(log)?;
        writeln_or!(log, "===== ===== ===== ===== result")?;
        let x = vec_y.rows(0 .. n);
        let lmd = vec_y.rows(n .. n + m);
        let nu = vec_y.rows(n + m .. n + m + p);
        writeln_or!(log, "x : {}", x.t())?;
        writeln_or!(log, "lmd : {}", lmd.t())?;
        writeln_or!(log, "nu : {}", nu.t())?;

        Ok(vec_y)
    }
}
