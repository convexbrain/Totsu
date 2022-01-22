//! First-order conic linear program solver

use num_traits::Float;
use core::marker::PhantomData;
use core::fmt::{Debug, Display, LowerExp};
use crate::linalg::LinAlg;
use crate::operator::Operator;
use crate::cone::Cone;
use crate::utils::*;

//

/// Solver errors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SolverError
{
    /// Found an unbounded certificate.
    Unbounded,
    /// Found an infeasibile certificate.
    Infeasible,
    /// Exceed max iterations.
    ExcessIter,

    /// Invalid [`Operator`].
    InvalidOp,
    /// Shortage of work slice length.
    WorkShortage,
    /// Failure caused by [`Cone`].
    ConeFailure,
}

impl Display for SolverError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", match &self {
            SolverError::Unbounded    => "Unbounded: found an unbounded certificate",
            SolverError::Infeasible   => "Infeasible: found an infeasibile certificate",
            SolverError::ExcessIter   => "ExcessIter: exceed max iterations",
            SolverError::InvalidOp    => "InvalidOp: invalid Operator",
            SolverError::WorkShortage => "WorkShortage: shortage of work slice length",
            SolverError::ConeFailure  => "ConeFailure: failure caused by Cone",
        })
    }
}

//

/// Solver parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct SolverParam<F: Float>
{
    /// Max iteration number of first-order algorithm. `None` means no upper limit.
    pub max_iter: Option<usize>,
    /// Tolerance of the primal residual, dual residual and duality gap.
    pub eps_acc: F,
    /// Tolerance of the unboundness and infeasibility.
    pub eps_inf: F,
    /// Tolerance of small positive value to avoid division by zero.
    pub eps_zero: F,
    /// Period of iterations to output progress log(for debug/trace level).
    pub log_period: usize,
}

impl<F: Float> Default for SolverParam<F>
{
    fn default() -> Self
    {
        let ten = F::from(10).unwrap();

        SolverParam {
            max_iter: None,
            eps_acc: ten.powi(-6),
            eps_inf: ten.powi(-6),
            eps_zero: ten.powi(-12),
            log_period: 10_000,
        }
    }
}

//

struct SelfDualEmbed<F: Float, L: LinAlg<F>, OC: Operator<F>, OA: Operator<F>, OB: Operator<F>>
{
    ph_f: PhantomData<F>,
    ph_l: PhantomData<L>,
    c: OC,
    a: OA,
    b: OB,
}

impl<F, L, OC, OA, OB> SelfDualEmbed<F, L, OC, OA, OB>
where F: Float, L: LinAlg<F>, OC: Operator<F>, OA: Operator<F>, OB: Operator<F>
{
    fn c(&self) -> &OC
    {
        &self.c
    }

    fn a(&self) -> &OA
    {
        &self.a
    }

    fn b(&self) -> &OB
    {
        &self.b
    }

    fn norm_b(&self,
        work_v: &mut[F], work_t: &mut[F]) -> F
    {
        Self::fr_norm(self.b(), work_v, work_t)
    }

    fn norm_c(&self,
        work_v: &mut[F], work_t: &mut[F]) -> F
    {
        Self::fr_norm(self.c(), work_v, work_t)
    }

    // Frobenius norm
    fn fr_norm<O: Operator<F>>(
        op: &O,
        work_v: &mut[F], work_t: &mut[F]) -> F
    {
        assert_eq!(work_v.len(), op.size().1);
        assert_eq!(work_t.len(), op.size().0);

        let f0 = F::zero();
        let f1 = F::one();

        L::scale(f0, work_v);
        let mut sq_norm = f0;

        for row in 0.. op.size().1 {
            work_v[row] = f1;
            op.op(f1, work_v, f0, work_t);
            let n = L::norm(work_t);
            sq_norm = sq_norm + n * n;
            work_v[row] = f0;
        }

        sq_norm.sqrt()
    }
    
    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (m, n) = self.a.size();
        
        assert_eq!(x.len(), n + m + m + 1);
        assert_eq!(y.len(), n + m + 1);

        let (x_x, x_y, x_s, x_tau) = x.split4(n, m, m, 1).unwrap();

        let (y_n, y_m, y_1) = y.split3(n, m, 1).unwrap();

        let f1 = F::one();

        self.a.trans_op(alpha, x_y, beta, y_n);
        self.c.op(alpha, x_tau, f1, y_n);

        self.a.op(-alpha, x_x, beta, y_m);
        L::add(-alpha, x_s, y_m);
        self.b.op(alpha, x_tau, f1, y_m);

        self.c.trans_op(-alpha, x_x, beta, y_1);
        self.b.trans_op(-alpha, x_y, f1, y_1);
    }

    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (m, n) = self.a.size();
        
        assert_eq!(x.len(), n + m + 1);
        assert_eq!(y.len(), n + m + m + 1);

        let (x_n, x_m, x_1) = x.split3(n, m, 1).unwrap();

        let (y_x, y_y, y_s, y_tau) = y.split4(n, m, m, 1).unwrap();

        let f1 = F::one();

        self.a.trans_op(-alpha, x_m, beta, y_x);
        self.c.op(-alpha, x_1, f1, y_x);

        self.a.op(alpha, x_n, beta, y_y);
        self.b.op(-alpha, x_1, f1, y_y);

        L::scale(beta, y_s);
        L::add(-alpha, x_m, y_s);

        self.c.trans_op(alpha, x_n, beta, y_tau);
        self.b.trans_op(alpha, x_m, f1, y_tau);
    }

    fn abssum_cols(&self, tau: &mut[F])
    {
        let (m, n) = self.a.size();
        let sz = (n + m + 1, n + m + m + 1);

        crate::operator::reffn::abssum_cols::<L, _, _>(
            sz,
            |x, y| self.op(F::one(), x, F::zero(), y),
            F::zero(), tau
        );
    }

    fn abssum_rows(&self, sigma: &mut[F])
    {
        let (m, n) = self.a.size();
        let sz = (n + m + 1, n + m + m + 1);

        crate::operator::reffn::abssum_rows::<L, _, _>(
            sz,
            |x, y| self.trans_op(F::one(), x, F::zero(), y),
            F::zero(), sigma
        );
    }
}

//

/// First-order conic linear program solver struct.
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
/// 
/// This struct abstracts a solver of a conic linear program:
/// \\[
/// \begin{array}{ll}
/// {\rm minimize} & c^T x \\\\
/// {\rm subject \ to} & A x + s = b \\\\
/// & s \in \mathcal{K},
/// \end{array}
/// \\]
/// where
/// * variables \\( x \in \mathbb{R}^n,\ s \in \mathbb{R}^m \\)
/// * \\( c \in \mathbb{R}^n \\) as an objective linear operator 
/// * \\( A \in \mathbb{R}^{m \times n} \\) and \\( b \in \mathbb{R}^m \\) as constraint linear operators 
/// * a nonempty, closed, convex cone \\( \mathcal{K} \\).
/// 
/// The solution gives optimal values of primal variables \\(x\\)
/// as well as dual variables \\(y\\) of the dual problem:
/// \\[
/// \begin{array}{ll}
/// {\rm maximize} & -b^T y \\\\
/// {\rm subject \ to} & -A^T y = c \\\\
/// & y \in \mathcal{K}^*,
/// \end{array}
/// \\]
/// where
/// * variables \\( y \in \mathbb{R}^m \\)
/// * \\( \mathcal{K}^* \\) is the dual cone of \\( \mathcal{K} \\).
pub struct Solver<L: LinAlg<F>, F: Float>
{
    /// solver parameters.
    pub par: SolverParam<F>,

    ph_l: PhantomData<L>,
}

impl<L, F> Solver<L, F>
where L: LinAlg<F>, F: Float
{
    /// Query of a length of work slice.
    /// 
    /// Returns a length of work slice that [`Solver::solve`] requires.
    /// * `op_a_size` is a number of rows and columns of \\(A\\).
    pub fn query_worklen(op_a_size: (usize, usize)) -> usize
    {
        let (m, n) = op_a_size;
    
        let len_iteration =
            n + m + m + 1 +  // x
            n + m + 1 +      // y
            n + m + m + 1 +  // dp_tau
            n + m + 1 +      // dp_sigma
            n + m + m + 1 +  // tmpw rx
            n + m + m + 1;   // tmpw tx
            //n + m + 1      // tmpw ty (share with tx)
            //m              // tmpw p (share with rx)
            //n              // tmpw d (share with rx)
        
        // len_norms = n.max(m) share with x
        
        len_iteration
    }

    /// Creates an instance.
    /// 
    /// Returns [`Solver`] instance.
    pub fn new() -> Self
    {
        Solver {
            par: SolverParam::default(),
            ph_l: PhantomData,
        }
    }

    /// Changes solver parameters.
    /// 
    /// Returns [`Solver`] with its parameters changed.
    /// * `f` is a function to change parameters given by its argument.
    pub fn par<P>(mut self, f: P) -> Self
    where P: FnOnce(&mut SolverParam<F>)
    {
        f(&mut self.par);
        self
    }
}

impl<L, F> Solver<L, F>
where L: LinAlg<F>, F: Float + Debug + LowerExp
{
    /// Starts to solve a conic linear program.
    /// 
    /// Returns `Ok` with a tuple of optimal \\(x, y\\)
    /// or `Err` with [`SolverError`] type.
    /// * `op_c` is \\(c\\) as a linear [`Operator`].
    /// * `op_a` is \\(A\\) as a linear [`Operator`].
    /// * `op_b` is \\(b\\) as a linear [`Operator`].
    /// * `cone` is \\(\mathcal{K}\\) expressed by [`Cone`].
    /// * `work` slice is used for temporal variables. [`Solver::solve`] does not rely on dynamic heap allocation.
    pub fn solve<OC, OA, OB, C>(self,
        (op_c, op_a, op_b, cone, work): (OC, OA, OB, C, &mut[F])
    ) -> Result<(&[F], &[F]), SolverError>
    where OC: Operator<F>, OA: Operator<F>, OB: Operator<F>, C: Cone<F>
    {
        let (m, n) = op_a.size();

        if op_c.size() != (n, 1) || op_b.size() != (m, 1) {
            log::error!("Size mismatch: op_c{:?}, op_a{:?}, op_b{:?}", op_c.size(), op_a.size(), op_b.size());
            return Err(SolverError::InvalidOp);
        }
    
        let op_k = SelfDualEmbed {
            ph_f: PhantomData::<F>,
            ph_l: PhantomData::<L>,
            c: op_c, a: op_a, b: op_b
        };

        let core = SolverCore {
            par: self.par,
            op_k,
            cone,
        };

        core.solve(work)
    }
}

//

struct SolverCore<L, F, OC, OA, OB, C>
where L: LinAlg<F>, F: Float + Debug + LowerExp,
      OC: Operator<F>, OA: Operator<F>, OB: Operator<F>, C: Cone<F>
{
    par: SolverParam<F>,

    op_k: SelfDualEmbed<F, L, OC, OA, OB>,
    cone: C,
}

impl<L, F, OC, OA, OB, C> SolverCore<L, F, OC, OA, OB, C>
where L: LinAlg<F>, F: Float + Debug + LowerExp,
      OC: Operator<F>, OA: Operator<F>, OB: Operator<F>, C: Cone<F>
{
    fn solve(mut self, work: &mut[F]) -> Result<(&[F], &[F]), SolverError>
    {
        log::info!("----- Initializing");
        let (m, n) = self.op_k.a().size();

        // Calculate norms
        let (norm_b, norm_c) = self.calc_norms(work)?;

        // Initialize vectors
        let (x, y, dp_tau, dp_sigma, tmpw) = self.init_vecs(work)?;

        // Calculate diagonal preconditioning
        self.calc_precond(dp_tau, dp_sigma);

        // Iteration
        log::info!("----- Started");
        let mut i = 0;
        loop {
            let excess_iter = if let Some(max_iter) = self.par.max_iter {
                i + 1 >= max_iter
            } else {
                false
            };

            let log_trig = if self.par.log_period > 0 {
                i % self.par.log_period == 0
            }
            else {
                if i == 0 && log::log_enabled!(log::Level::Debug) {
                    log::warn!("log_period == 0: no periodic log");
                }
                false
            };

            // Update vectors
            let val_tau = self.update_vecs(x, y, dp_tau, dp_sigma, tmpw)?;

            if log_trig {
                log::trace!("{}: state {:?} {:?}", i, x, y);
            }

            if val_tau > self.par.eps_zero {
                // Termination criteria of convergence
                let (cri_pri, cri_dual, cri_gap) = self.criteria_conv(x, norm_c, norm_b, tmpw);

                let term_conv = (cri_pri <= self.par.eps_acc) && (cri_dual <= self.par.eps_acc) && (cri_gap <= self.par.eps_acc);

                if log_trig || excess_iter || term_conv {
                    log::debug!("{}: pri_dual_gap {:.2e} {:.2e} {:.2e}", i, cri_pri, cri_dual, cri_gap);
                }

                if excess_iter || term_conv {
                    let (x_x_ast, x_y_ast) = x.split2(n, m).unwrap();
                    L::scale(val_tau.recip(), x_x_ast);
                    L::scale(val_tau.recip(), x_y_ast);

                    log::trace!("{}: x {:?}", i, x_x_ast);
                    log::trace!("{}: y {:?}", i, x_y_ast);

                    if term_conv {
                        log::info!("----- Converged");

                        return Ok((x_x_ast, x_y_ast));
                    }
                    else {
                        log::warn!("----- ExcessIter");

                        return Err(SolverError::ExcessIter);
                    }
                }
            }
            else {
                // Termination criteria of infeasibility
                let (cri_unbdd, cri_infeas) = self.criteria_inf(x, norm_c, norm_b, tmpw);

                let term_unbdd = cri_unbdd <= self.par.eps_inf;
                let term_infeas = cri_infeas <= self.par.eps_inf;

                if log_trig || excess_iter || term_unbdd || term_infeas {
                    log::debug!("{}: unbdd_infeas {:.2e} {:.2e}", i, cri_unbdd, cri_infeas);
                }

                if excess_iter || term_unbdd || term_infeas {
                    let (x_x_cert, x_y_cert) = x.split2(n, m).unwrap();

                    log::trace!("{}: x {:?}", i, x_x_cert);
                    log::trace!("{}: y {:?}", i, x_y_cert);

                    if term_unbdd {
                        log::warn!("----- Unbounded");

                        return Err(SolverError::Unbounded);
                    }
                    else if term_infeas {
                        log::warn!("----- Infeasible");

                        return Err(SolverError::Infeasible);
                    }
                    else {
                        log::warn!("----- ExcessIter");

                        return Err(SolverError::ExcessIter);
                    }
                }
            }

            i += 1;
            assert!(!excess_iter);
        } // end of loop
    }

    fn calc_norms(&mut self, work: &mut[F])
    -> Result<(F, F), SolverError>
    {
        let work_one = &mut [F::zero()];
        
        let norm_b = {
            let (m, _) = self.op_k.b().size();
            let t = work.split1(m).ok_or(SolverError::WorkShortage)?;
    
            self.op_k.norm_b(work_one, t)
        };
    
        let norm_c = {
            let (n, _) = self.op_k.c().size();
            let t = work.split1(n).ok_or(SolverError::WorkShortage)?;
    
            self.op_k.norm_c(work_one, t)
        };
    
        Ok((norm_b, norm_c))
    }
    
    fn init_vecs<'b>(&self, work: &'b mut[F])
    -> Result<(&'b mut[F], &'b mut[F], &'b mut[F], &'b mut[F], &'b mut[F]), SolverError>
    {
        let (m, n) = self.op_k.a().size();

        let (x, y, dp_tau, dp_sigma, tmpw) = work.split5(
            n + m + m + 1,
            n + m + 1,
            n + m + m + 1,
            n + m + 1,
            (n + m + m + 1) * 2,
        ).ok_or(SolverError::WorkShortage)?;

        let f0 = F::zero();
        let f1 = F::one();

        L::scale(f0, x);
        L::scale(f0, y);
    
        x[n + m + m] = f1; // x_tau
    
        Ok((x, y, dp_tau, dp_sigma, tmpw))
    }

    fn calc_precond(&self, dp_tau: &mut[F], dp_sigma: &mut[F])
    {
        let (m, n) = self.op_k.a().size();

        log::info!("----- 0");
        self.op_k.abssum_cols(dp_tau);
        log::info!("----- 1");
        for tau in dp_tau.iter_mut() {
            *tau = tau.max(self.par.eps_zero).recip();
        }

        log::info!("----- 2");
        self.op_k.abssum_rows(dp_sigma);
        log::info!("----- 3");
        for sigma in dp_sigma.iter_mut() {
            *sigma = sigma.max(self.par.eps_zero).recip();
        }

        // grouping dependent on cone
        let group = |tau_group: &mut[F]| {
            if tau_group.len() > 0 {
                let mut min_t = tau_group[0];
                for t in tau_group.iter() {
                    min_t = min_t.min(*t);
                }
                for t in tau_group.iter_mut() {
                    *t = min_t;
                }
            }
        };
        let (_, dpt_dual_cone, dpt_cone, _) = dp_tau.split4(n, m, m, 1).unwrap();
        self.cone.product_group(dpt_dual_cone, group);
        self.cone.product_group(dpt_cone, group);
    }

    fn update_vecs(&mut self, x: &mut[F], y: &mut[F], dp_tau: &[F], dp_sigma: &[F], tmpw: &mut[F])
    -> Result<F, SolverError>
    {
        let (m, n) = self.op_k.a().size();

        let (rx, tx) = tmpw.split2(x.len(), x.len()).unwrap();

        let val_tau;

        let f0 = F::zero();
        let f1 = F::one();
    
        L::copy(x, rx); // rx := x_k

        { // Update x := x_{k+1} before projection
            self.op_k.trans_op(-f1, y, f0, tx);
            L::transform_di(f1, dp_tau, tx, f1, x);
        }

        { // Projection prox_G(x)
            let (_x_x, x_y, x_s, x_tau) = x.split4(n, m, m, 1).unwrap();

            self.cone.proj(true, x_y).or(Err(SolverError::ConeFailure))?;
            self.cone.proj(false, x_s).or(Err(SolverError::ConeFailure))?;
            x_tau[0] = x_tau[0].max(f0);

            val_tau = x_tau[0];
        }

        L::add(-f1-f1, x, rx); // rx := x_k - 2 * x_{k+1}

        { // Update y := y_{k+1} before projection
            let ty = tx.split1(y.len()).unwrap();
            
            self.op_k.op(-f1, rx, f0, ty);
            L::transform_di(f1, dp_sigma, ty, f1, y);
        }

        { // Projection prox_F*(y)
            let (y_1, _) = y.split_last_mut().unwrap();

            *y_1 = y_1.min(f0);
        }

        Ok(val_tau)
    }

    fn criteria_conv(&self, x: &[F], norm_c: F, norm_b: F, tmpw: &mut[F])
    -> (F, F, F)
    {
        let (m, n) = self.op_k.a().size();

        let (x_x, x_y, x_s, x_tau) = x.split4(n, m, m, 1).unwrap();
        let (p, d) = tmpw.split2(m, n).unwrap();
    
        let f0 = F::zero();
        let f1 = F::one();
    
        let val_tau = x_tau[0];
        assert!(val_tau > f0);
    
        let work_one = &mut [f1];
    
        // Calc convergence criteria
        
        L::copy(x_s, p);
        self.op_k.b().op(-f1, work_one, val_tau.recip(), p);
        self.op_k.a().op(val_tau.recip(), x_x, f1, p);
    
        self.op_k.c().op(f1, work_one, f0, d);
        self.op_k.a().trans_op(val_tau.recip(), x_y, f1, d);
    
        self.op_k.c().trans_op(val_tau.recip(), x_x, f0, work_one);
        let g_x = work_one[0];
    
        self.op_k.b().trans_op(val_tau.recip(), x_y, f0, work_one);
        let g_y = work_one[0];
    
        let g = g_x + g_y;
    
        let cri_pri = L::norm(p) / (f1 + norm_b);
        let cri_dual = L::norm(d) / (f1 + norm_c);
        let cri_gap = g.abs() / (f1 + g_x.abs() + g_y.abs());
    
        (cri_pri, cri_dual, cri_gap)
    }
    
    fn criteria_inf(&self, x: &[F], norm_c: F, norm_b: F, tmpw: &mut[F])
    -> (F, F)
    {
        let (m, n) = self.op_k.a().size();

        let (x_x, x_y, x_s, _) = x.split4(n, m, m, 1).unwrap();
        let (p, d) = tmpw.split2(m, n).unwrap();

        let f0 = F::zero();
        let f1 = F::one();
        let finf = F::infinity();
    
        let work_one = &mut [f0];

        // Calc undoundness and infeasibility criteria
        
        L::copy(x_s, p);
        self.op_k.a().op(f1, x_x, f1, p);

        self.op_k.a().trans_op(f1, x_y, f0, d);

        self.op_k.c().trans_op(-f1, x_x, f0, work_one);
        let m_cx = work_one[0];

        self.op_k.b().trans_op(-f1, x_y, f0, work_one);
        let m_by = work_one[0];

        let cri_unbdd = if m_cx > self.par.eps_zero {
            L::norm(p) * norm_c / m_cx
        }
        else {
            finf
        };
        let cri_infeas = if m_by > self.par.eps_zero {
            L::norm(d) * norm_b / m_by
        }
        else {
            finf
        };

        (cri_unbdd, cri_infeas)
    }
}
