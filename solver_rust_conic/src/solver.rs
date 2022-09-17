//! First-order conic linear program solver

use num_traits::{Float, Zero, One};
use core::marker::PhantomData;
use core::fmt::{Debug, Display, LowerExp};
use crate::linalg::{SliceLike, LinAlg};
use crate::operator::Operator;
use crate::cone::Cone;
use crate::{splitm, splitm_mut};

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

struct SelfDualEmbed<L, OC, OA, OB>
where L: LinAlg, OC: Operator<L>, OA: Operator<L>, OB: Operator<L>
{
    ph_l: PhantomData<L>,
    c: OC,
    a: OA,
    b: OB,
}

impl<L, OC, OA, OB> SelfDualEmbed<L, OC, OA, OB>
where L: LinAlg, OC: Operator<L>, OA: Operator<L>, OB: Operator<L>
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
        work_v: &mut L::Sl, work_t: &mut L::Sl) -> L::F
    {
        Self::fr_norm(self.b(), work_v, work_t)
    }

    fn norm_c(&self,
        work_v: &mut L::Sl, work_t: &mut L::Sl) -> L::F
    {
        Self::fr_norm(self.c(), work_v, work_t)
    }

    // Frobenius norm
    fn fr_norm<O: Operator<L>>(
        op: &O,
        work_v: &mut L::Sl, work_t: &mut L::Sl) -> L::F
    {
        assert_eq!(work_v.len(), op.size().1);
        assert_eq!(work_t.len(), op.size().0);

        let f0 = L::F::zero();
        let f1 = L::F::one();

        L::scale(f0, work_v);
        let mut sq_norm = f0;

        for row in 0.. op.size().1 {
            work_v.set(row, f1);
            op.op(f1, work_v, f0, work_t);
            let n = L::norm(work_t);
            sq_norm = sq_norm + n * n;
            work_v.set(row, f0);
        }

        sq_norm.sqrt()
    }
    
    fn op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        let (m, n) = self.a.size();
        
        assert_eq!(x.len(), n + m + m + 1);
        assert_eq!(y.len(), n + m + 1);

        splitm!(x, (x_x; n), (x_y; m), (x_s; m), (x_tau; 1));

        splitm_mut!(y, (y_n; n), (y_m; m), (y_1; 1));

        let f1 = L::F::one();

        self.a.trans_op(alpha, &x_y, beta, &mut y_n);
        self.c.op(alpha, &x_tau, f1, &mut y_n);

        self.a.op(-alpha, &x_x, beta, &mut y_m);
        L::add(-alpha, &x_s, &mut y_m);
        self.b.op(alpha, &x_tau, f1, &mut y_m);

        self.c.trans_op(-alpha, &x_x, beta, &mut y_1);
        self.b.trans_op(-alpha, &x_y, f1, &mut y_1);
    }

    fn trans_op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        let (m, n) = self.a.size();
        
        assert_eq!(x.len(), n + m + 1);
        assert_eq!(y.len(), n + m + m + 1);

        splitm!(x, (x_n; n), (x_m; m), (x_1; 1));

        splitm_mut!(y, (y_x; n), (y_y; m), (y_s; m), (y_tau; 1));

        let f1 = L::F::one();

        self.a.trans_op(-alpha, &x_m, beta, &mut y_x);
        self.c.op(-alpha, &x_1, f1, &mut y_x);

        self.a.op(alpha, &x_n, beta, &mut y_y);
        self.b.op(-alpha, &x_1, f1, &mut y_y);

        L::scale(beta, &mut y_s);
        L::add(-alpha, &x_m, &mut y_s);

        self.c.trans_op(alpha, &x_n, beta, &mut y_tau);
        self.b.trans_op(alpha, &x_m, f1, &mut y_tau);
    }

    fn abssum(&self, tau: &mut L::Sl, sigma: &mut L::Sl)
    {
        let (m, n) = self.a.size();
        let f0 = L::F::zero();
        let f1 = L::F::one();

        L::scale(f0, tau);

        splitm_mut!(tau, (tau_x; n), (tau_y; m), (tau_s; m), (tau_tau; 1));

        self.a.absadd_cols(&mut tau_x);
        self.c.absadd_rows(&mut tau_x);
        self.a.absadd_rows(&mut tau_y);
        self.b.absadd_rows(&mut tau_y);
        L::adds(f1, &mut tau_s);
        self.c.absadd_cols(&mut tau_tau);
        self.b.absadd_cols(&mut tau_tau);

        splitm_mut!(sigma, (sigma_n; n), (sigma_m; m), (sigma_1; 1));

        L::copy(&tau_x, &mut sigma_n);
        L::copy(&tau_y, &mut sigma_m);
        L::add(f1, &tau_s, &mut sigma_m);
        L::copy(&tau_tau, &mut sigma_1);
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
pub struct Solver<L: LinAlg>
{
    /// solver parameters.
    pub par: SolverParam<L::F>,
}

impl<L: LinAlg> Solver<L>
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
        }
    }

    /// Changes solver parameters.
    /// 
    /// Returns [`Solver`] with its parameters changed.
    /// * `f` is a function to change parameters given by its argument.
    pub fn par<P>(mut self, f: P) -> Self
    where P: FnOnce(&mut SolverParam<L::F>)
    {
        f(&mut self.par);
        self
    }
}

impl<L: LinAlg> Solver<L>
where L::F: Float + Debug + LowerExp
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
        (op_c, op_a, op_b, cone, work): (OC, OA, OB, C, &mut[L::F])
    ) -> Result<(&[L::F], &[L::F]), SolverError>
    where OC: Operator<L>, OA: Operator<L>, OB: Operator<L>, C: Cone<L>
    {
        let (m, n) = op_a.size();

        if op_c.size() != (n, 1) || op_b.size() != (m, 1) {
            log::error!("Size mismatch: op_c{:?}, op_a{:?}, op_b{:?}", op_c.size(), op_a.size(), op_b.size());
            return Err(SolverError::InvalidOp);
        }
    
        if Self::query_worklen((m, n)) > work.len() {
            log::error!("Work memory length {} must be >= {}", work.len(), Self::query_worklen((m, n)));
            return Err(SolverError::WorkShortage);
        }

        log::debug!("{:?}", self.par);

        let op_k = SelfDualEmbed {
            ph_l: PhantomData::<L>,
            c: op_c, a: op_a, b: op_b
        };

        let core = SolverCore {
            par: self.par,
            op_k,
            cone,
        };

        let rslt = core.solve(&mut L::Sl::new_mut(work));

        let (sol_x, rest) = work.split_at(n);
        let (sol_y, _) = rest.split_at(m);
        
        rslt.map(|_| {(sol_x, sol_y)})
    }
}

//

struct SolverCore<L, OC, OA, OB, C>
where L: LinAlg, L::F: Float + Debug + LowerExp,
      OC: Operator<L>, OA: Operator<L>, OB: Operator<L>, C: Cone<L>
{
    par: SolverParam<L::F>,

    op_k: SelfDualEmbed<L, OC, OA, OB>,
    cone: C,
}

impl<L, OC, OA, OB, C> SolverCore<L, OC, OA, OB, C>
where L: LinAlg, L::F: Float + Debug + LowerExp,
      OC: Operator<L>, OA: Operator<L>, OB: Operator<L>, C: Cone<L>
{
    fn solve(mut self, work: &mut L::Sl) -> Result<(), SolverError>
    {
        log::info!("----- Initializing");
        let (m, n) = self.op_k.a().size();

        // Calculate norms
        let (norm_b, norm_c) = self.calc_norms(work);

        // Initialize vectors
        splitm_mut!(work,
            (x; n + m + m + 1),
            (y; n + m + 1),
            (dp_tau; n + m + m + 1),
            (dp_sigma; n + m + 1),
            (tmpw; (n + m + m + 1) * 2)
        );
        self.init_vecs(&mut x, &mut y);

        // Calculate diagonal preconditioning
        self.calc_precond(&mut dp_tau, &mut dp_sigma);

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
            let val_tau = self.update_vecs(&mut x, &mut y, &dp_tau, &dp_sigma, &mut tmpw)?;

            if val_tau > self.par.eps_zero {
                // Termination criteria of convergence
                let (cri_pri, cri_dual, cri_gap) = self.criteria_conv(&x, norm_c, norm_b, &mut tmpw);

                let term_conv = (cri_pri <= self.par.eps_acc) && (cri_dual <= self.par.eps_acc) && (cri_gap <= self.par.eps_acc);

                if log_trig || excess_iter || term_conv {
                    log::debug!("{}: pri_dual_gap {:.2e} {:.2e} {:.2e}", i, cri_pri, cri_dual, cri_gap);
                }
                else {
                    log::trace!("{}: pri_dual_gap {:.2e} {:.2e} {:.2e}", i, cri_pri, cri_dual, cri_gap);
                }

                if excess_iter || term_conv {
                    splitm_mut!(x, (x_x_ast; n), (x_y_ast; m));
                    L::scale(val_tau.recip(), &mut x_x_ast);
                    L::scale(val_tau.recip(), &mut x_y_ast);

                    log::trace!("{}: x {:?}", i, x_x_ast.get_ref());
                    log::trace!("{}: y {:?}", i, x_y_ast.get_ref());

                    if term_conv {
                        log::info!("----- Converged");

                        return Ok(());
                    }
                    else {
                        log::warn!("----- ExcessIter");

                        return Err(SolverError::ExcessIter);
                    }
                }
            }
            else {
                // Termination criteria of infeasibility
                let (cri_unbdd, cri_infeas) = self.criteria_inf(&x, norm_c, norm_b, &mut tmpw);

                let term_unbdd = cri_unbdd <= self.par.eps_inf;
                let term_infeas = cri_infeas <= self.par.eps_inf;

                if log_trig || excess_iter || term_unbdd || term_infeas {
                    log::debug!("{}: unbdd_infeas {:.2e} {:.2e}", i, cri_unbdd, cri_infeas);
                }
                else {
                    log::trace!("{}: unbdd_infeas {:.2e} {:.2e}", i, cri_unbdd, cri_infeas);
                }

                if excess_iter || term_unbdd || term_infeas {
                    splitm!(x, (x_x_cert; n), (x_y_cert; m));

                    log::trace!("{}: x {:?}", i, x_x_cert.get_ref());
                    log::trace!("{}: y {:?}", i, x_y_cert.get_ref());

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

    fn calc_norms(&mut self, work: &mut L::Sl)
    -> (L::F, L::F)
    {
        let mut work1 = [L::F::zero()];
        let mut work_one = L::Sl::new_mut(&mut work1);
        
        let norm_b = {
            let (m, _) = self.op_k.b().size();
            splitm_mut!(work, (t; m));
    
            self.op_k.norm_b(&mut work_one, &mut t)
        };
    
        let norm_c = {
            let (n, _) = self.op_k.c().size();
            splitm_mut!(work, (t; n));
    
            self.op_k.norm_c(&mut work_one, &mut t)
        };
    
        (norm_b, norm_c)
    }

    fn init_vecs(&self, x: &mut L::Sl, y: &mut L::Sl)
    {
        let (m, n) = self.op_k.a().size();

        let f0 = L::F::zero();
        let f1 = L::F::one();

        L::scale(f0, x);
        L::scale(f0, y);
    
        x.set(n + m + m, f1); // x_tau
    }

    fn calc_precond(&self, dp_tau: &mut L::Sl, dp_sigma: &mut L::Sl)
    {
        let (m, n) = self.op_k.a().size();

        self.op_k.abssum(dp_tau, dp_sigma);
        for tau in dp_tau.get_mut() {
            *tau = (*tau).max(self.par.eps_zero).recip();
        }
        for sigma in dp_sigma.get_mut() {
            *sigma = (*sigma).max(self.par.eps_zero).recip();
        }

        // grouping dependent on cone
        let group = |tau_group: &mut L::Sl| {
            if tau_group.len() > 0 {
                let tau_group_mut = tau_group.get_mut();
                let mut min_t = tau_group_mut[0];
                for t in tau_group_mut.iter() {
                    min_t = min_t.min(*t);
                }
                for t in tau_group_mut.iter_mut() {
                    *t = min_t;
                }
            }
        };
        splitm_mut!(dp_tau, (_dpt_n; n), (dpt_dual_cone; m), (dpt_cone; m), (_dpt_1; 1));
        self.cone.product_group(&mut dpt_dual_cone, group);
        self.cone.product_group(&mut dpt_cone, group);
    }

    fn update_vecs(&mut self, x: &mut L::Sl, y: &mut L::Sl, dp_tau: &L::Sl, dp_sigma: &L::Sl, tmpw: &mut L::Sl)
    -> Result<L::F, SolverError>
    {
        let (m, n) = self.op_k.a().size();

        splitm_mut!(tmpw, (rx; x.len()), (tx; x.len()));

        let val_tau;

        let f0 = L::F::zero();
        let f1 = L::F::one();
    
        L::copy(x, &mut rx); // rx := x_k

        { // Update x := x_{k+1} before projection
            self.op_k.trans_op(-f1, y, f0, &mut tx);
            L::transform_di(f1, dp_tau, &tx, f1, x);
        }

        { // Projection prox_G(x)
            splitm_mut!(x, (_x_x; n), (x_y; m), (x_s; m), (x_tau; 1));

            self.cone.proj(true, &mut x_y).or(Err(SolverError::ConeFailure))?;
            self.cone.proj(false, &mut x_s).or(Err(SolverError::ConeFailure))?;

            val_tau = x_tau.get(0).max(f0);
            x_tau.set(0, val_tau);
        }

        L::add(-f1-f1, x, &mut rx); // rx := x_k - 2 * x_{k+1}

        { // Update y := y_{k+1} before projection
            splitm_mut!(tx, (ty; y.len()));
            self.op_k.op(-f1, &rx, f0, &mut ty);
            L::transform_di(f1, dp_sigma, &ty, f1, y);
        }

        { // Projection prox_F*(y)
            splitm_mut!(y, (_y_nm; n + m), (y_1; 1));

            let kappa = y_1.get(0).min(f0);
            y_1.set(0, kappa);
        }

        Ok(val_tau)
    }

    fn criteria_conv(&self, x: &L::Sl, norm_c: L::F, norm_b: L::F, tmpw: &mut L::Sl)
    -> (L::F, L::F, L::F)
    {
        let (m, n) = self.op_k.a().size();

        splitm!(x, (x_x; n), (x_y; m), (x_s; m), (x_tau; 1));
        splitm_mut!(tmpw, (p; m), (d; n));
    
        let f0 = L::F::zero();
        let f1 = L::F::one();
    
        let val_tau = x_tau.get(0);
        assert!(val_tau > f0);
    
        let mut work1 = [f1];
        let mut work_one = L::Sl::new_mut(&mut work1);
    
        // Calc convergence criteria
        
        L::copy(&x_s, &mut p);
        self.op_k.b().op(-f1, &work_one, val_tau.recip(), &mut p);
        self.op_k.a().op(val_tau.recip(), &x_x, f1, &mut p);
    
        self.op_k.c().op(f1, &work_one, f0, &mut d);
        self.op_k.a().trans_op(val_tau.recip(), &x_y, f1, &mut d);
    
        self.op_k.c().trans_op(val_tau.recip(), &x_x, f0, &mut work_one);
        let g_x = work_one.get(0);
    
        self.op_k.b().trans_op(val_tau.recip(), &x_y, f0, &mut work_one);
        let g_y = work_one.get(0);
    
        let g = g_x + g_y;
    
        let cri_pri = L::norm(&p) / (f1 + norm_b);
        let cri_dual = L::norm(&d) / (f1 + norm_c);
        let cri_gap = g.abs() / (f1 + g_x.abs() + g_y.abs());
    
        (cri_pri, cri_dual, cri_gap)
    }
    
    fn criteria_inf(&self, x: &L::Sl, norm_c: L::F, norm_b: L::F, tmpw: &mut L::Sl)
    -> (L::F, L::F)
    {
        let (m, n) = self.op_k.a().size();

        splitm!(x, (x_x; n), (x_y; m), (x_s; m), (_x_tau; 1));
        splitm_mut!(tmpw, (p; m), (d; n));

        let f0 = L::F::zero();
        let f1 = L::F::one();
        let finf = L::F::infinity();
    
        let mut work1 = [f0];
        let mut work_one = L::Sl::new_mut(&mut work1);

        // Calc undoundness and infeasibility criteria
        
        L::copy(&x_s, &mut p);
        self.op_k.a().op(f1, &x_x, f1, &mut p);

        self.op_k.a().trans_op(f1, &x_y, f0, &mut d);

        self.op_k.c().trans_op(-f1, &x_x, f0, &mut work_one);
        let m_cx = work_one.get(0);

        self.op_k.b().trans_op(-f1, &x_y, f0, &mut work_one);
        let m_by = work_one.get(0);

        let cri_unbdd = if m_cx > self.par.eps_zero {
            L::norm(&p) * norm_c / m_cx
        }
        else {
            finf
        };
        let cri_infeas = if m_by > self.par.eps_zero {
            L::norm(&d) * norm_b / m_by
        }
        else {
            finf
        };

        (cri_unbdd, cri_infeas)
    }
}
