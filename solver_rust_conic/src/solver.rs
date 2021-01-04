// TODO: no-std

use num::Float;
use core::marker::PhantomData;
use core::fmt::{Debug, LowerExp};

pub trait LinAlg<F: Float>
{
    fn norm(x: &[F]) -> F;
    fn inner_prod(x: &[F], y: &[F]) -> F;
    fn copy(x: &[F], y: &mut[F]);
    fn scale(alpha: F, x: &mut[F]);
    fn add(alpha: F, x: &[F], y: &mut[F]);
}

pub trait Operator<F: Float>
{
    fn size(&self) -> (usize, usize);

    // y = alpha * Op * x + beta * y
    // x.len() shall be size().1
    // y.len() shall be size().0
    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F]);

    // y = alpha * Op^T * x + beta * y
    // x.len() shall be size().0
    // y.len() shall be size().1
    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F]);
}

pub trait Cone<F: Float>
{
    // x.len() shall be op_a.size().0
    fn proj(&mut self, par: &SolverParam<F>, x: &mut[F]) -> Result<(), SolverError>;
    fn dual_proj(&mut self, par: &SolverParam<F>, x: &mut[F]) -> Result<(), SolverError>
    {
        self.proj(par, x) // Self-dual cone
    }
}

macro_rules! writeln_or {
    ( $( $arg: expr ),* ) => {
        writeln!( $( $arg ),* ).or(Err(SolverError::LogFailure))
    };
}

#[derive(Debug)]
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

#[derive(Debug)]
pub struct SolverParam<F: Float>
{
    pub max_iter: Option<usize>,
    pub eps_acc: F,
    pub eps_inf: F,
    pub eps_zero: F,
    pub log_period: usize,
    pub log_verbose: bool,
}

impl<F: Float> Default for SolverParam<F>
{
    fn default() -> Self
    {
        let ten = F::from(10).unwrap();

        SolverParam {
            max_iter: Some(10_000),
            eps_acc: ten.powi(-6),
            eps_inf: ten.powi(-6),
            eps_zero: ten.powi(-12),
            log_period: 0,
            log_verbose: false,
        }
    }
}

fn split_tup6<T>(
    s: &[T], pos: (usize, usize, usize, usize, usize, usize)
) -> Result<(&[T], &[T], &[T], &[T], &[T], &[T]), SolverError>
{
    if s.len() < pos.0 + pos.1 + pos.2 + pos.3 + pos.4 + pos.5 {
        Err(SolverError::WorkShortage)
    }
    else {
        let (s0, spl) = s.split_at(pos.0);
        let (s1, spl) = spl.split_at(pos.1);
        let (s2, spl) = spl.split_at(pos.2);
        let (s3, spl) = spl.split_at(pos.3);
        let (s4, spl) = spl.split_at(pos.4);
        let (s5, _) = spl.split_at(pos.5);

        Ok((s0, s1, s2, s3, s4, s5))
    }
}

fn split_tup6_mut<T>(
    s: &mut[T], pos: (usize, usize, usize, usize, usize, usize)
) -> Result<(&mut[T], &mut[T], &mut[T], &mut[T], &mut[T], &mut[T]), SolverError>
{
    if s.len() < pos.0 + pos.1 + pos.2 + pos.3 + pos.4 {
        Err(SolverError::WorkShortage)
    }
    else {
        let (s0, spl) = s.split_at_mut(pos.0);
        let (s1, spl) = spl.split_at_mut(pos.1);
        let (s2, spl) = spl.split_at_mut(pos.2);
        let (s3, spl) = spl.split_at_mut(pos.3);
        let (s4, spl) = spl.split_at_mut(pos.4);
        let (s5, _) = spl.split_at_mut(pos.5);

        Ok((s0, s1, s2, s3, s4, s5))
    }
}

struct SelfDualEmbed<F: Float, L: LinAlg<F>, OC: Operator<F>, OA: Operator<F>, OB: Operator<F>>
{
    _ph_f: PhantomData<F>,
    _ph_l: PhantomData<L>,
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

    // spectral norm
    fn sp_norm(&self,
        eps_zero: F,
        work_v: &mut[F], work_t: &mut[F], work_w: &mut[F]) -> F
    {
        assert_eq!(work_v.len(), self.size().1);
        assert_eq!(work_t.len(), self.size().0);
        assert_eq!(work_w.len(), self.size().1);

        for e in work_v.iter_mut() {
            *e = F::one();
        }
        let mut lambda = F::zero();

        loop {
            let n = L::norm(work_v);
            if n > F::min_positive_value() {
                L::scale(n.recip(), work_v);
            }

            self.op(F::one(), work_v, F::zero(), work_t);
            self.trans_op(F::one(), work_t, F::zero(), work_w);

            let lambda_n = L::inner_prod(work_v, work_w);

            if (lambda_n - lambda).abs() <= eps_zero {
                return lambda_n.sqrt();
            }

            L::copy(work_w, work_v);
            lambda = lambda_n;
        }
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

        for e in work_v.iter_mut() {
            *e = F::zero();
        }
        let mut sq_norm = F::zero();

        for row in 0.. op.size().1 {
            work_v[row] = F::one();
            op.op(F::one(), work_v, F::zero(), work_t);
            let n = L::norm(work_t);
            sq_norm = sq_norm + n * n;
            work_v[row] = F::zero();
        }

        sq_norm.sqrt()
    }
    
    fn size(&self) -> (usize, usize)
    {
        let (m, n) = self.a.size();
        let nm1 = n + m + 1;

        (nm1, nm1 * 2)
    }

    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (m, n) = self.a.size();
        
        let full_x = if x.len() == (n + m + 1) * 2 {
            true
        }
        else if x.len() == n + (m + 1) * 2 {
            false
        }
        else {
            assert!(false);
            false
        };
        assert_eq!(y.len(), n + m + 1);

        let (u_x, u_y, u_tau, v_r, v_s, v_kappa) = split_tup6(x, (n, m, 1, if full_x {n} else {0}, m, 1)).unwrap();

        let (w_n, w_m, w_1, _, _, _) = split_tup6_mut(y, (n, m, 1, 0, 0, 0)).unwrap();

        self.a.trans_op(alpha, u_y, beta, w_n);
        self.c.op(alpha, u_tau, F::one(), w_n);
        if full_x {
            L::add(-alpha, v_r, w_n);
        }

        self.a.op(-alpha, u_x, beta, w_m);
        self.b.op(alpha, u_tau, F::one(), w_m);
        L::add(-alpha, v_s, w_m);

        self.c.trans_op(-alpha, u_x, beta, w_1);
        self.b.trans_op(-alpha, u_y, F::one(), w_1);
        L::add(-alpha, v_kappa, w_1);
    }

    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (m, n) = self.a.size();
        
        assert_eq!(x.len(), n + m + 1);
        let full_y = if y.len() == (n + m + 1) * 2 {
            true
        }
        else if y.len() == n + (m + 1) * 2 {
            false
        }
        else {
            assert!(false);
            false
        };

        let (w_n, w_m, w_1, _, _, _) = split_tup6(x, (n, m, 1, 0, 0, 0)).unwrap();

        let (u_x, u_y, u_tau, v_r, v_s, v_kappa) = split_tup6_mut(y, (n, m, 1, if full_y {n} else {0}, m, 1)).unwrap();

        self.a.trans_op(-alpha, w_m, beta, u_x);
        self.c.op(-alpha, w_1, F::one(), u_x);

        self.a.op(alpha, w_n, beta, u_y);
        self.b.op(-alpha, w_1, F::one(), u_y);

        self.c.trans_op(alpha, w_n, beta, u_tau);
        self.b.trans_op(alpha, w_m, F::one(), u_tau);

        if full_y {
            L::scale(beta, v_r);
            L::add(-alpha, w_n, v_r);
        }

        L::scale(beta, v_s);
        L::add(-alpha, w_m, v_s);

        L::scale(beta, v_kappa);
        L::add(-alpha, w_1, v_kappa);
    }
}

pub struct Solver<F: Float + Debug + LowerExp, L: LinAlg<F>, W: core::fmt::Write>
{
    pub par: SolverParam<F>,

    _ph_l: PhantomData::<L>,
    logger: W,
}

impl<F, L, W> Solver<F, L, W>
where F: Float + Debug + LowerExp, L: LinAlg<F>, W: core::fmt::Write
{
    pub fn query_worklen(&self, op_a_size: (usize, usize)) -> usize
    {
        let (m, n) = op_a_size;

        let len_norms = (n + m + 1) * 5;
        let len_iteration = (n + (m + 1) * 2) * 2 + (n + m + 1) + n + m;

        len_norms.max(len_iteration)
    }

    pub fn new(_linalg: L, logger: W) -> Self
    {
        Solver {
            par: SolverParam::default(),
            _ph_l: PhantomData::<L>,
            logger,
        }
    }

    pub fn solve<OC, OA, OB, C>(self,
        op_c: OC, op_a: OA, op_b: OB, cone: C,
        work: &mut[F],
    ) -> Result<(&[F], &[F]), SolverError>
    where OC: Operator<F>, OA: Operator<F>, OB: Operator<F>, C: Cone<F>
    {
        let (m, n) = op_a.size();

        if op_c.size() != (n, 1) || op_b.size() != (m, 1) {
            return Err(SolverError::InvalidOp);
        }
    
        let op_l = SelfDualEmbed {
            _ph_f: PhantomData::<F>,
            _ph_l: PhantomData::<L>,
            c: op_c, a: op_a, b: op_b
        };

        let core = SolverCore {
            par: self.par,
            _ph_l: PhantomData,
            logger: self.logger,
            op_l,
            cone,
        };

        core.solve(work)
    }
}

struct SolverCore<
    F: Float + Debug + LowerExp, L: LinAlg<F>, W: core::fmt::Write,
    OC: Operator<F>, OA: Operator<F>, OB: Operator<F>, C: Cone<F>
    >
{
    par: SolverParam<F>,

    _ph_l: PhantomData<L>,
    logger: W,

    op_l: SelfDualEmbed<F, L, OC, OA, OB>,
    cone: C,
}

impl<F, L, W, OC, OA, OB, C> SolverCore<F, L, W, OC, OA, OB, C>
where F: Float + Debug + LowerExp, L: LinAlg<F>, W: core::fmt::Write,
      OC: Operator<F>, OA: Operator<F>, OB: Operator<F>, C: Cone<F>
{
    fn solve(mut self, work: &mut[F]) -> Result<(&[F], &[F]), SolverError>
    {
        writeln_or!(self.logger, "----- Started")?;
        let (m, n) = self.op_l.a().size();

        // Calculate norms
        let (op_l_norm, norm_b, norm_c) = self.calc_norms(work)?;

        let tau = op_l_norm.recip();
        let sigma = op_l_norm.recip();

        // Initialize vectors
        let (x, y, xx, p, d) = self.init_vecs(work)?;

        let mut i = 0;
        loop {
            let over_iter = if let Some(max_iter) = self.par.max_iter {
                i + 1 >= max_iter
            } else {
                false
            };

            let log_trig = if self.par.log_period > 0 {
                i % self.par.log_period == 0
            }
            else {
                false
            };

            // Update vectors
            let u_tau = self.update_vecs(x, y, xx, tau, sigma)?;

            if log_trig && self.par.log_verbose {
                writeln_or!(self.logger, "{}: state {:?} {:?}", i, x, y)?;
            }

            if u_tau > self.par.eps_zero {
                // Termination criteria of convergence
                let (cri_pri, cri_dual, cri_gap) = self.criteria_conv(x, p, d, norm_c, norm_b);

                let term_conv = (cri_pri <= self.par.eps_acc) && (cri_dual <= self.par.eps_acc) && (cri_gap <= self.par.eps_acc);

                if log_trig || over_iter || term_conv {
                    writeln_or!(self.logger, "{}: pri_dual_gap {:.2e} {:.2e} {:.2e}", i, cri_pri, cri_dual, cri_gap)?;
                }

                if over_iter || term_conv {
                    let (u_x_ast, u_y_ast, _, _, _, _) = split_tup6_mut(x, (n, m, 0, 0, 0, 0)).unwrap();
                    L::scale(u_tau.recip(), u_x_ast);
                    L::scale(u_tau.recip(), u_y_ast);

                    if self.par.log_verbose {
                        writeln_or!(self.logger, "{}: x {:?}", i, u_x_ast)?;
                        writeln_or!(self.logger, "{}: y {:?}", i, u_y_ast)?;
                    }

                    if term_conv {
                        writeln_or!(self.logger, "----- Converged")?;

                        return Ok((u_x_ast, u_y_ast));
                    }
                    else {
                        writeln_or!(self.logger, "----- OverIter")?;

                        return Err(SolverError::OverIter);
                    }
                }
            }
            else {
                // Termination criteria of infeasibility
                let (cri_unbdd, cri_infeas) = self.criteria_inf(x, p, d, norm_c, norm_b);

                let term_unbdd = cri_unbdd <= self.par.eps_inf;
                let term_infeas = cri_infeas <= self.par.eps_inf;

                if log_trig || over_iter || term_unbdd || term_infeas {
                    writeln_or!(self.logger, "{}: unbdd_infeas {:.2e} {:.2e}", i, cri_unbdd, cri_infeas)?;
                }

                if over_iter || term_unbdd || term_infeas {
                    let (u_x_cert, u_y_cert, _, _, _, _) = split_tup6(x, (n, m, 0, 0, 0, 0)).unwrap();

                    if self.par.log_verbose {
                        writeln_or!(self.logger, "{}: x {:?}", i, u_x_cert)?;
                        writeln_or!(self.logger, "{}: y {:?}", i, u_y_cert)?;
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
    -> Result<(F, F, F), SolverError>
    {
        let work_one = &mut [F::zero()];
        
        let op_l_norm = {
            let (nrow, ncol) = self.op_l.size();
            let (v, t, w, _, _, _) = split_tup6_mut(work, (ncol, nrow, ncol, 0, 0, 0))?;
    
            self.op_l.sp_norm(self.par.eps_zero, v, t, w)
        };
        if op_l_norm < F::min_positive_value() {
            return Err(SolverError::InvalidOp);
        }
    
        let norm_b = {
            let (nrow, _) = self.op_l.b().size();
            let (t, _, _, _, _, _) = split_tup6_mut(work, (nrow, 0, 0, 0, 0, 0))?;
    
            self.op_l.norm_b(work_one, t)
        };
    
        let norm_c = {
            let (nrow, _) = self.op_l.c().size();
            let (t, _, _, _, _, _) = split_tup6_mut(work, (nrow, 0, 0, 0, 0, 0))?;
    
            self.op_l.norm_c(work_one, t)
        };
    
        Ok((op_l_norm, norm_b, norm_c))
    }
    
    fn init_vecs<'b>(&self, work: &'b mut[F])
    -> Result<(&'b mut[F], &'b mut[F], &'b mut[F], &'b mut[F], &'b mut[F]), SolverError>
    {
        let (m, n) = self.op_l.a().size();

        let (x, y, xx, p, d, _) = split_tup6_mut(work, (
            n + (m + 1) * 2,
            n + m + 1,
            n + (m + 1) * 2,
            m,
            n,
            0
        ))?;
    
        for e in x.iter_mut() {
            *e = F::zero();
        }
        for e in y.iter_mut() {
            *e = F::zero();
        }
        x[n + m] = F::one(); // u_tau
        x[n + m + 1 + m] = F::one(); // v_kappa
    
        L::copy(x, xx);
    
        Ok((x, y, xx, p, d))
    }
    
    fn update_vecs(&mut self,
        x: &mut[F], y: &mut[F], xx: &mut[F],
        tau: F, sigma: F)
    -> Result<F, SolverError>
    {
        let (m, n) = self.op_l.a().size();

        let ret_u_tau;

        self.op_l.trans_op(-tau, y, F::one(), x);

        { // Projection
            let (_, u_y, u_tau, v_s, v_kappa, _) = split_tup6_mut(x, (n, m, 1, m, 1, 0)).unwrap();

            self.cone.dual_proj(&self.par, u_y)?;
            u_tau[0] = u_tau[0].max(F::zero());
            self.cone.proj(&self.par, v_s)?;
            v_kappa[0] = v_kappa[0].max(F::zero());

            ret_u_tau = u_tau[0];
        }

        L::add(-F::one()-F::one(), x, xx);
        self.op_l.op(-sigma, xx, F::one(), y);
        L::copy(x, xx);

        Ok(ret_u_tau)
    }

    fn criteria_conv(&self,
        x: &[F], p: &mut[F], d: &mut[F],
        norm_c: F, norm_b: F)
    -> (F, F, F)
    {
        let (m, n) = self.op_l.a().size();

        let (u_x, u_y, u_tau, v_s, _, _) = split_tup6(x, (n, m, 1, m, 0, 0)).unwrap();
    
        let u_tau = u_tau[0];
        assert!(u_tau > F::zero());
    
        let work_one = &mut [F::one()];
    
        // Calc convergence criteria
        
        L::copy(v_s, p);
        self.op_l.b().op(-F::one(), work_one, u_tau.recip(), p);
        self.op_l.a().op(u_tau.recip(), u_x, F::one(), p);
    
        self.op_l.c().op(F::one(), work_one, F::zero(), d);
        self.op_l.a().trans_op(u_tau.recip(), u_y, F::one(), d);
    
        self.op_l.c().trans_op(u_tau.recip(), u_x, F::zero(), work_one);
        let g_x = work_one[0];
    
        self.op_l.b().trans_op(u_tau.recip(), u_y, F::zero(), work_one);
        let g_y = work_one[0];
    
        let g = g_x + g_y;
    
        let cri_pri = L::norm(p) / (F::one() + norm_b);
        let cri_dual = L::norm(d) / (F::one() + norm_c);
        let cri_gap = g.abs() / (F::one() + g_x.abs() + g_y.abs());
    
        (cri_pri, cri_dual, cri_gap)
    }
    
    fn criteria_inf(&self,
        x: &[F], p: &mut[F], d: &mut[F],
        norm_c: F, norm_b: F)
    -> (F, F)
    {
        let (m, n) = self.op_l.a().size();

        let (u_x, u_y, _, v_s, _, _) = split_tup6(x, (n, m, 1, m, 0, 0)).unwrap();

        let work_one = &mut [F::zero()];

        // Calc undoundness and infeasibility criteria
        
        L::copy(v_s, p);
        self.op_l.a().op(F::one(), u_x, F::one(), p);

        self.op_l.a().trans_op(F::one(), u_y, F::zero(), d);

        self.op_l.c().trans_op(-F::one(), u_x, F::zero(), work_one);
        let m_cx = work_one[0];

        self.op_l.b().trans_op(-F::one(), u_y, F::zero(), work_one);
        let m_by = work_one[0];

        let cri_unbdd = if m_cx > self.par.eps_zero {
            L::norm(p) * norm_c / m_cx
        }
        else {
            F::infinity()
        };
        let cri_infeas = if m_by > self.par.eps_zero {
            L::norm(d) * norm_b / m_by
        }
        else {
            F::infinity()
        };

        (cri_unbdd, cri_infeas)
    }
}
