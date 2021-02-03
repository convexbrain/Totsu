use num::Float;
use core::marker::PhantomData;
use core::fmt::{Debug, LowerExp};
use crate::linalg::LinAlg;
use crate::operator::Operator;
use crate::cone::Cone;
use crate::utils::*;

//

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SolverError
{
    Unbounded,
    Infeasible,
    OverIter,
    OverIterInf,

    InvalidOp,
    WorkShortage,
    LogFailure,
    ConeFailure,
}

//

#[derive(Debug, Clone, PartialEq)]
pub struct SolverParam<F: Float>
{
    pub max_iter: Option<usize>,
    pub eps_acc: F,
    pub eps_inf: F,
    pub eps_zero: F,
    pub log_period: Option<usize>,
    pub log_verbose: bool,
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
            log_period: None,
            log_verbose: false,
        }
    }
}

//

macro_rules! writeln_or {
    ( $( $arg: expr ),* ) => {
        writeln!( $( $arg ),* ).or(Err(SolverError::LogFailure))
    };
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
}

//

pub struct Solver<L: LinAlg<F>, F: Float>
{
    pub par: SolverParam<F>,

    ph_l: PhantomData<L>,
}

impl<L, F> Solver<L, F>
where L: LinAlg<F>, F: Float
{
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
            //n + m + m + 1  // tmpw wy (share with rx)
            //n + m + 1      // tmpw wy (share with tx)
            //m              // tmpw p (share with rx)
            //n              // tmpw d (share with rx)
        
        // len_norms = n.max(m) share with x
        
        len_iteration
    }

    pub fn new() -> Self
    {
        Solver {
            par: SolverParam::default(),
            ph_l: PhantomData,
        }
    }

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
    pub fn solve<OC, OA, OB, C, W>(self,
        (op_c, op_a, op_b, cone, work): (OC, OA, OB, C, &mut[F]),
        logger: W
    ) -> Result<(&[F], &[F]), SolverError>
    where OC: Operator<F>, OA: Operator<F>, OB: Operator<F>, C: Cone<F>, W: core::fmt::Write
    {
        let (m, n) = op_a.size();

        if op_c.size() != (n, 1) || op_b.size() != (m, 1) {
            return Err(SolverError::InvalidOp);
        }
    
        let op_k = SelfDualEmbed {
            ph_f: PhantomData::<F>,
            ph_l: PhantomData::<L>,
            c: op_c, a: op_a, b: op_b
        };

        let core = SolverCore {
            par: self.par,
            logger,
            op_k,
            cone,
        };

        core.solve(work)
    }
}

//

struct SolverCore<L, F, OC, OA, OB, C, W>
where L: LinAlg<F>, F: Float + Debug + LowerExp,
      OC: Operator<F>, OA: Operator<F>, OB: Operator<F>, C: Cone<F>,
      W: core::fmt::Write
{
    par: SolverParam<F>,

    logger: W,

    op_k: SelfDualEmbed<F, L, OC, OA, OB>,
    cone: C,
}

impl<L, F, OC, OA, OB, C, W> SolverCore<L, F, OC, OA, OB, C, W>
where L: LinAlg<F>, F: Float + Debug + LowerExp,
      OC: Operator<F>, OA: Operator<F>, OB: Operator<F>, C: Cone<F>,
      W: core::fmt::Write
{
    fn solve(mut self, work: &mut[F]) -> Result<(&[F], &[F]), SolverError>
    {
        writeln_or!(self.logger, "----- Started")?;
        let (m, n) = self.op_k.a().size();

        // Calculate norms
        let (norm_b, norm_c) = self.calc_norms(work)?;

        // Initialize vectors
        let (x, y, dp_tau, dp_sigma, tmpw) = self.init_vecs(work)?;

        // Calculate diagonal preconditioning
        self.calc_precond(dp_tau, dp_sigma, tmpw);

        // Iteration
        let mut i = 0;
        loop {
            let over_iter = if let Some(max_iter) = self.par.max_iter {
                i + 1 >= max_iter
            } else {
                false
            };

            let log_trig = if let Some(log_period) = self.par.log_period {
                i % log_period.max(1) == 0
            }
            else {
                false
            };

            // Update vectors
            let val_tau = self.update_vecs(x, y, dp_tau, dp_sigma, tmpw)?;

            if log_trig && self.par.log_verbose {
                writeln_or!(self.logger, "{}: state {:?} {:?}", i, x, y)?;
            }

            if val_tau > self.par.eps_zero {
                // Termination criteria of convergence
                let (cri_pri, cri_dual, cri_gap) = self.criteria_conv(x, norm_c, norm_b, tmpw);

                let term_conv = (cri_pri <= self.par.eps_acc) && (cri_dual <= self.par.eps_acc) && (cri_gap <= self.par.eps_acc);

                if log_trig || over_iter || term_conv {
                    writeln_or!(self.logger, "{}: pri_dual_gap {:.2e} {:.2e} {:.2e}", i, cri_pri, cri_dual, cri_gap)?;
                }

                if over_iter || term_conv {
                    let (x_x_ast, x_y_ast) = x.split2(n, m).unwrap();
                    L::scale(val_tau.recip(), x_x_ast);
                    L::scale(val_tau.recip(), x_y_ast);

                    if self.par.log_verbose {
                        writeln_or!(self.logger, "{}: x {:?}", i, x_x_ast)?;
                        writeln_or!(self.logger, "{}: y {:?}", i, x_y_ast)?;
                    }

                    if term_conv {
                        writeln_or!(self.logger, "----- Converged")?;

                        return Ok((x_x_ast, x_y_ast));
                    }
                    else {
                        writeln_or!(self.logger, "----- OverIter")?;

                        return Err(SolverError::OverIter);
                    }
                }
            }
            else {
                // Termination criteria of infeasibility
                let (cri_unbdd, cri_infeas) = self.criteria_inf(x, norm_c, norm_b, tmpw);

                let term_unbdd = cri_unbdd <= self.par.eps_inf;
                let term_infeas = cri_infeas <= self.par.eps_inf;

                if log_trig || over_iter || term_unbdd || term_infeas {
                    writeln_or!(self.logger, "{}: unbdd_infeas {:.2e} {:.2e}", i, cri_unbdd, cri_infeas)?;
                }

                if over_iter || term_unbdd || term_infeas {
                    let (x_x_cert, x_y_cert) = x.split2(n, m).unwrap();

                    if self.par.log_verbose {
                        writeln_or!(self.logger, "{}: x {:?}", i, x_x_cert)?;
                        writeln_or!(self.logger, "{}: y {:?}", i, x_y_cert)?;
                    }

                    if term_unbdd {
                        writeln_or!(self.logger, "----- Unbounded")?;

                        return Err(SolverError::Unbounded);
                    }
                    else if term_infeas {
                        writeln_or!(self.logger, "----- Infeasible")?;

                        return Err(SolverError::Infeasible);
                    }
                    else {
                        writeln_or!(self.logger, "----- OverIterInf")?;

                        return Err(SolverError::OverIterInf);
                    }
                }
            }

            i += 1;
            assert!(!over_iter);
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

    fn calc_precond(&self, dp_tau: &mut[F], dp_sigma: &mut[F], tmpw: &mut[F])
    {
        let (m, n) = self.op_k.a().size();
        let (wx, wy) = tmpw.split2(dp_tau.len(), dp_sigma.len()).unwrap();

        let f0 = F::zero();
        let f1 = F::one();

        L::scale(f0, wx);
        for (i, tau) in dp_tau.iter_mut().enumerate() {
            wx[i] = f1;
            self.op_k.op(f1, wx, f0, wy);
            let asum = L::abssum(wy);
            *tau = asum.max(self.par.eps_zero).recip();
            wx[i] = f0;
        }

        L::scale(f0, wy);
        for (i, sigma) in dp_sigma.iter_mut().enumerate() {
            wy[i] = f1;
            self.op_k.trans_op(f1, wy, f0, wx);
            let asum = L::abssum(wx);
            *sigma = asum.max(self.par.eps_zero).recip();
            wy[i] = f0;
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

            self.cone.proj(true, self.par.eps_zero, x_y).or(Err(SolverError::ConeFailure))?;
            self.cone.proj(false, self.par.eps_zero, x_s).or(Err(SolverError::ConeFailure))?;
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
