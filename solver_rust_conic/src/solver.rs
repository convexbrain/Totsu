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

pub trait Projection
{
    fn cone(&mut self, x: &mut[f64]);
    fn cone_conj(&mut self, x: &mut[f64])
    {
        self.cone(x); // Self-dual cone
    }
}

// spectral norm
fn sp_norm<O: Operator>(op: &O) -> f64
{
    // TODO: param
    let eps_zero = 1e-12;

    // TODO: memory
    let mut v = vec![1.; op.size().1];
    let mut t = vec![0.; op.size().0];
    let mut w = vec![0.; op.size().1];

    let mut lambda = 0.;

    loop {
        let n = norm(&v);
        if n > f64::MIN_POSITIVE {
            scale(n.recip(), &mut v);
        }

        op.op(1., &v, 0., &mut t);
        op.trans_op(1., &t, 0., &mut w);

        let lambda_n = inner_prod(&v, &w);

        if (lambda_n - lambda).abs() <= eps_zero {
            return lambda_n.sqrt();
        }

        copy(&w, &mut v);
        lambda = lambda_n;
    }
}

// Frobenius norm
fn fr_norm<O: Operator>(op: &O) -> f64
{
    // TODO: memory
    let mut v = vec![0.; op.size().1];
    let mut t = vec![0.; op.size().0];

    let mut sq_norm = 0.;

    for row in 0.. v.len() {
        v[row] = 1.;
        op.op(1., &v, 0., &mut t);
        let n = norm(&t);
        sq_norm += n * n;
        v[row] = 0.;
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

pub struct SolverParam
{
    max_iter: Option<usize>,
    eps_acc: f64,
    eps_inf: f64,
    eps_zero: f64,
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

pub struct Solver;

impl Solver
{
    pub fn query_worklen(op_a_size: (usize, usize)) -> usize
    {
        let (m, n) = op_a_size;

        let len_norms = (n + m + 1) * 5;
        let len_iteration = (n + (m + 1) * 2) * 2 + (n + m + 1) + n + m;

        len_norms.max(len_iteration)
    }

    pub fn solve<OC, OA, OB, PJ>(
        par: SolverParam,
        op_c: OC, op_a: OA, op_b: OB, mut proj: PJ
    )
    where OC: Operator, OA: Operator, OB: Operator, PJ: Projection
    {
        let (m, n) = op_a.size();

        let op_l = SelfDualEmbed::new(op_c, op_a, op_b);
        let op_l_norm = sp_norm(&op_l);
        // TODO: error
        assert!(op_l_norm >= f64::MIN_POSITIVE);

        let tau = op_l_norm.recip();
        let sigma = op_l_norm.recip();

        let b_norm = fr_norm(op_l.b());
        let c_norm = fr_norm(op_l.c());

        // TODO: memory
        let mut x = vec![0.; n + (m + 1) * 2];
        x[n + m] = 1.; // u_tau
        x[n + m + 1 + m] = 1.; // v_kappa
        let mut xx = x.clone();
        let mut y = vec![0.; n + m + 1];

        // TODO: memory
        let mut p = vec![0.; m];
        let mut d = vec![0.; n];
        let one = &mut [1.];

        let mut i = 0;
        loop {
            // TODO: log
            //println!("----- {}", i);

            { // Update iteration
                op_l.trans_op(-tau, &y, 1.0, &mut x);

                { // Projection
                    let (_, u) = x.split_at_mut(n);
                    let (u_y, u) = u.split_at_mut(m);
                    let (u_tau, v) = u.split_at_mut(1);
                    let (v_s, v) = v.split_at_mut(m);
                    let (v_kappa, _) = v.split_at_mut(1);
    
                    proj.cone_conj(u_y);
                    u_tau[0] = u_tau[0].max(0.);
                    proj.cone(v_s);
                    v_kappa[0] = v_kappa[0].max(0.);
                }
    
                add(-2., &x, &mut xx);
                op_l.op(-sigma, &xx, 1., &mut y);
                copy(&x, &mut xx);
            }

            { // Termination criteria
                let (u_x, u) = x.split_at(n);
                let (u_y, u) = u.split_at(m);
                let (u_tau, v) = u.split_at(1);
                let (v_s, _) = v.split_at(m);

                let u_tau = u_tau[0];

                if u_tau > par.eps_zero {
                    // Check convergence

                    one[0] = 1.;

                    copy(v_s, &mut p);
                    op_l.b().op(-1., one, u_tau.recip(), &mut p);
                    op_l.a().op(u_tau.recip(), u_x, 1., &mut p);
    
                    op_l.c().op(1., one, 0., &mut d);
                    op_l.a().trans_op(u_tau.recip(), u_y, 1., &mut d);
    
                    op_l.c().trans_op(u_tau.recip(), u_x, 0., one);
                    let g_x = one[0];
    
                    op_l.b().trans_op(u_tau.recip(), u_y, 0., one);
                    let g_y = one[0];
    
                    let g = g_x + g_y;

                    let term_pri = norm(&p) <= par.eps_acc * (1. + b_norm);
                    let term_dual = norm(&d) <= par.eps_acc * (1. + c_norm);
                    let term_gap = g.abs() <= par.eps_acc * (1. + g_x.abs() + g_y.abs());
    
                    // TODO: log
                    //println!("{} {} {}", term_pri, term_dual, term_gap);

                    if term_pri && term_dual && term_gap {
                        // TODO: log
                        println!("converged");
                        scale(u_tau.recip(), &mut x);
                        println!("{:?}", x);
                        // TODO: result
                        return;
                    }
                }
                else {
                    // Check undoundness and infeasibility
                    
                    copy(v_s, &mut p);
                    op_l.a().op(1., u_x, 1., &mut p);

                    op_l.a().trans_op(1., u_y, 0., &mut d);

                    op_l.c().trans_op(-1., u_x, 0., one);
                    let m_cx = one[0];

                    op_l.b().trans_op(-1., u_y, 0., one);
                    let m_by = one[0];
        
                    let term_unbdd = (m_cx > par.eps_zero) && (
                        norm(&p) * c_norm <= par.eps_inf * m_cx
                    );
        
                    let term_infeas = (m_by > par.eps_zero) && (
                        norm(&d) * b_norm <= par.eps_inf * m_by
                    );
        
                    // TODO: log
                    //println!("{} {}", term_unbdd, term_infeas);
        
                    if term_unbdd {
                        // TODO: log
                        println!("unbounded");
                        // TODO: result
                        return;
                    }
        
                    if term_infeas {
                        // TODO: log
                        println!("infeasible");
                        // TODO: result
                        return;
                    }
                }
            }

            i += 1;
            if let Some(max_iter) = par.max_iter {
                if i >= max_iter {
                    // TODO: log
                    println!("timeover");
                    // TODO: result
                    return;
                }
            }
        } // end of loop
    }
}
