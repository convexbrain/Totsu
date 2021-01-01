// TODO: num-float
// TODO: doc

use crate::linalg::{norm, scale, inner_prod, copy, add};

pub trait Operator
{
    fn size(&self) -> (usize, usize);
    // y = alpha * Op * x + beta * y
    fn op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64]);
    // y = alpha * Op^T * x + beta * y
    fn trans_op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64]);
}

pub trait Cone
{
    fn proj(&mut self, par: &SolverParam, x: &mut[f64]);
    fn dual_proj(&mut self, par: &SolverParam, x: &mut[f64])
    {
        self.proj(par, x); // Self-dual cone
    }
}

// spectral norm
fn sp_norm<O: Operator>(
    op: &O, eps_zero: f64,
    work_v: &mut[f64], work_t: &mut[f64], work_w: &mut[f64]) -> f64
{
    assert_eq!(work_v.len(), op.size().1);
    assert_eq!(work_t.len(), op.size().0);
    assert_eq!(work_w.len(), op.size().1);

    for e in work_v.iter_mut() {
        *e = 1.;
    }
    let mut lambda = 0.;

    loop {
        let n = norm(work_v);
        if n > f64::MIN_POSITIVE {
            scale(n.recip(), work_v);
        }

        op.op(1., work_v, 0., work_t);
        op.trans_op(1., work_t, 0., work_w);

        let lambda_n = inner_prod(work_v, work_w);

        if (lambda_n - lambda).abs() <= eps_zero {
            return lambda_n.sqrt();
        }

        copy(work_w, work_v);
        lambda = lambda_n;
    }
}

// Frobenius norm
fn fr_norm<O: Operator>(
    op: &O,
    work_v: &mut[f64], work_t: &mut[f64]) -> f64
{
    assert_eq!(work_v.len(), op.size().1);
    assert_eq!(work_t.len(), op.size().0);

    for e in work_v.iter_mut() {
        *e = 0.;
    }
    let mut sq_norm = 0.;

    for row in 0.. op.size().1 {
        work_v[row] = 1.;
        op.op(1., work_v, 0., work_t);
        let n = norm(work_t);
        sq_norm += n * n;
        work_v[row] = 0.;
    }

    sq_norm.sqrt()
}

struct SelfDualEmbed<OC: Operator, OA: Operator, OB: Operator>
{
    n: usize,
    m: usize,
    c: OC,
    a: OA,
    b: OB,
}

impl<OC, OA, OB> SelfDualEmbed<OC, OA, OB>
where OC: Operator, OA: Operator, OB: Operator
{
    fn new(c: OC, a: OA, b: OB) -> Self
    {
        let (m, n) = a.size();

        // TODO: error
        assert_eq!(c.size(), (n, 1));
        assert_eq!(b.size(), (m, 1));

        SelfDualEmbed {
            n, m, c, a, b
        }
    }

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
}

impl<OC, OA, OB> Operator for SelfDualEmbed<OC, OA, OB>
where OC: Operator, OA: Operator, OB: Operator
{
    fn size(&self) -> (usize, usize)
    {
        let nm1 = self.n + self.m + 1;

        (nm1, nm1 * 2)
    }

    fn op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        let n = self.n;
        let m = self.m;
        
        assert_eq!(y.len(), n + m + 1);

        let (u_x, u) = x.split_at(n);
        let (u_y, u) = u.split_at(m);
        let (u_tau, v) = u.split_at(1);
        let (v_r, v) = if v.len() == n + m + 1 {
            v.split_at(n)
        }
        else if v.len() == m + 1 {
            v.split_at(0)
        }
        else {
            assert!(false);
            v.split_at(0)
        };
        let (v_s, v) = v.split_at(m);
        let (v_kappa, _) = v.split_at(1);

        let (w_n, w) = y.split_at_mut(n);
        let (w_m, w) = w.split_at_mut(m);
        let (w_1, _) = w.split_at_mut(1);

        self.a.trans_op(alpha, u_y, beta, w_n);
        self.c.op(alpha, u_tau, 1., w_n);
        if v_r.len() == w_n.len() {
            add(-alpha, v_r, w_n);
        }

        self.a.op(-alpha, u_x, beta, w_m);
        self.b.op(alpha, u_tau, 1., w_m);
        add(-alpha, v_s, w_m);

        self.c.trans_op(-alpha, u_x, beta, w_1);
        self.b.trans_op(-alpha, u_y, 1., w_1);
        add(-alpha, v_kappa, w_1);
    }

    fn trans_op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        let n = self.n;
        let m = self.m;
        
        assert_eq!(x.len(), n + m + 1);

        let (w_n, w) = x.split_at(n);
        let (w_m, w) = w.split_at(m);
        let (w_1, _) = w.split_at(1);

        let (u_x, u) = y.split_at_mut(n);
        let (u_y, u) = u.split_at_mut(m);
        let (u_tau, v) = u.split_at_mut(1);
        let (v_r, v) = if v.len() == n + m + 1 {
            v.split_at_mut(n)
        }
        else if v.len() == m + 1 {
            v.split_at_mut(0)
        }
        else {
            assert!(false);
            v.split_at_mut(0)
        };
        let (v_s, v) = v.split_at_mut(m);
        let (v_kappa, _) = v.split_at_mut(1);

        self.a.trans_op(-alpha, w_m, beta, u_x);
        self.c.op(-alpha, w_1, 1., u_x);

        self.a.op(alpha, w_n, beta, u_y);
        self.b.op(-alpha, w_1, 1., u_y);

        self.c.trans_op(alpha, w_n, beta, u_tau);
        self.b.trans_op(alpha, w_m, 1., u_tau);

        if v_r.len() == w_n.len() {
            scale(beta, v_r);
            add(-alpha, w_n, v_r);
        }

        scale(beta, v_s);
        add(-alpha, w_m, v_s);

        scale(beta, v_kappa);
        add(-alpha, w_1, v_kappa);
    }
}

#[derive(Debug)]
pub struct SolverParam
{
    pub max_iter: Option<usize>,
    pub eps_acc: f64,
    pub eps_inf: f64,
    pub eps_zero: f64,
}

impl Default for SolverParam
{
    fn default() -> Self
    {
        SolverParam {
            max_iter: Some(10_000),
            eps_acc: 1e-6,
            eps_inf: 1e-6,
            eps_zero: 1e-12,
        }
    }
}

#[derive(Debug)]
pub enum SolverError
{
    Unbounded,
    Infeasible,
    OverIter,
    OverIterInf,
}

pub fn query_worklen(op_a_size: (usize, usize)) -> usize
{
    let (m, n) = op_a_size;

    let len_norms = (n + m + 1) * 5;
    let len_iteration = (n + (m + 1) * 2) * 2 + (n + m + 1) + n + m;

    len_norms.max(len_iteration)
}

pub fn solve<OC, OA, OB, C>(
    par: SolverParam,
    op_c: OC, op_a: OA, op_b: OB, mut cone: C,
    work: &mut[f64]
) -> Result<(&[f64], &[f64]), SolverError>
where OC: Operator, OA: Operator, OB: Operator, C: Cone
{
    let (m, n) = op_a.size();
    let work_one = &mut [0.];

    let op_l = SelfDualEmbed::new(op_c, op_a, op_b);

    let op_l_norm = {
        // TODO: error
        let (v, spl_work) = work.split_at_mut(op_l.size().1);
        let (t, spl_work) = spl_work.split_at_mut(op_l.size().0);
        let (w, _) = spl_work.split_at_mut(op_l.size().1);

        sp_norm(&op_l, par.eps_zero, v, t, w)
    };
    // TODO: error
    assert!(op_l_norm >= f64::MIN_POSITIVE);

    let tau = op_l_norm.recip();
    let sigma = op_l_norm.recip();

    let b_norm = {
        let (t, _) = work.split_at_mut(op_l.b().size().0);

        fr_norm(op_l.b(), work_one, t)
    };

    let c_norm = {
        let (t, _) = work.split_at_mut(op_l.c().size().0);

        fr_norm(op_l.c(), work_one, t)
    };

    //

    // TODO: error
    let (x, spl_work) = work.split_at_mut(n + (m + 1) * 2);
    let (y, spl_work) = spl_work.split_at_mut(n + m + 1);
    let (xx, spl_work) = spl_work.split_at_mut(n + (m + 1) * 2);
    let (p, spl_work) = spl_work.split_at_mut(m);
    let (d, _) = spl_work.split_at_mut(n);
    for e in x.iter_mut() {
        *e = 0.;
    }
    for e in y.iter_mut() {
        *e = 0.;
    }
    for e in xx.iter_mut() {
        *e = 0.;
    }
    x[n + m] = 1.; // u_tau
    x[n + m + 1 + m] = 1.; // v_kappa
    xx[n + m] = 1.; // u_tau
    xx[n + m + 1 + m] = 1.; // v_kappa

    let mut i = 0;
    loop {
        // TODO: log
        //println!("----- {}", i);

        { // Update iteration
            op_l.trans_op(-tau, y, 1.0, x);

            { // Projection
                let (_, u) = x.split_at_mut(n);
                let (u_y, u) = u.split_at_mut(m);
                let (u_tau, v) = u.split_at_mut(1);
                let (v_s, v) = v.split_at_mut(m);
                let (v_kappa, _) = v.split_at_mut(1);

                cone.dual_proj(&par, u_y);
                u_tau[0] = u_tau[0].max(0.);
                cone.proj(&par, v_s);
                v_kappa[0] = v_kappa[0].max(0.);
            }

            add(-2., x, xx);
            op_l.op(-sigma, xx, 1., y);
            copy(x, xx);
        }

        i += 1;
        let over_iter = if let Some(max_iter) = par.max_iter {
            i >= max_iter
        } else {
            false
        };

        { // Termination criteria
            let (u_x, u) = x.split_at(n);
            let (u_y, u) = u.split_at(m);
            let (u_tau, v) = u.split_at(1);
            let (v_s, _) = v.split_at(m);

            let u_tau = u_tau[0];

            if u_tau > par.eps_zero {
                // Check convergence

                work_one[0] = 1.;

                copy(v_s, p);
                op_l.b().op(-1., work_one, u_tau.recip(), p);
                op_l.a().op(u_tau.recip(), u_x, 1., p);

                op_l.c().op(1., work_one, 0., d);
                op_l.a().trans_op(u_tau.recip(), u_y, 1., d);

                op_l.c().trans_op(u_tau.recip(), u_x, 0., work_one);
                let g_x = work_one[0];

                op_l.b().trans_op(u_tau.recip(), u_y, 0., work_one);
                let g_y = work_one[0];

                let g = g_x + g_y;

                let term_pri = norm(p) <= par.eps_acc * (1. + b_norm);
                let term_dual = norm(d) <= par.eps_acc * (1. + c_norm);
                let term_gap = g.abs() <= par.eps_acc * (1. + g_x.abs() + g_y.abs());

                // TODO: log
                //println!("{} {} {}", term_pri, term_dual, term_gap);

                if over_iter || (term_pri && term_dual && term_gap) {
                    let (u_x_ast, u) = x.split_at_mut(n);
                    let (u_y_ast, _) = u.split_at_mut(m);
                    scale(u_tau.recip(), u_x_ast);
                    scale(u_tau.recip(), u_y_ast);

                    if term_pri && term_dual && term_gap {
                        // TODO: log
                        //println!("converged");
                        //println!("{:?}", x_ast);
                        //println!("{:?}", y_ast);

                        return Ok((u_x_ast, u_y_ast));
                    }
                    else {
                        // TODO: log
                        //println!("overiter");

                        return Err(SolverError::OverIter);
                    }
                }
            }
            else {
                // Check undoundness and infeasibility
                
                copy(v_s, p);
                op_l.a().op(1., u_x, 1., p);

                op_l.a().trans_op(1., u_y, 0., d);

                op_l.c().trans_op(-1., u_x, 0., work_one);
                let m_cx = work_one[0];

                op_l.b().trans_op(-1., u_y, 0., work_one);
                let m_by = work_one[0];
    
                let term_unbdd = (m_cx > par.eps_zero) && (
                    norm(p) * c_norm <= par.eps_inf * m_cx
                );
    
                let term_infeas = (m_by > par.eps_zero) && (
                    norm(d) * b_norm <= par.eps_inf * m_by
                );
    
                // TODO: log
                //println!("{} {}", term_unbdd, term_infeas);

                if over_iter || term_unbdd || term_infeas {
                    let (u_x_cert, u) = x.split_at_mut(n);
                    let (u_y_cert, _) = u.split_at_mut(m);
                    if term_unbdd {
                        scale(m_cx.recip(), u_x_cert);
                    }
                    if term_infeas {
                        scale(m_by.recip(), u_y_cert);
                    }
    
                    if term_unbdd {
                        // TODO: log
                        //println!("unbounded");

                        return Err(SolverError::Unbounded);
                    }
                    else if term_infeas {
                        // TODO: log
                        //println!("infeasible");

                        return Err(SolverError::Infeasible);
                    }
                    else {
                        // TODO: log
                        //println!("overiterinf");

                        return Err(SolverError::OverIterInf);
                    }
                }
            }
        }

        assert!(!over_iter);
    } // end of loop
}
