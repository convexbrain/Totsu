use std::marker::PhantomData;
use num_traits::{Float, Zero, One, cast};
use totsu_core::solver::{Solver, SliceLike, Operator, Cone};
use totsu_core::{LinAlgEx, MatOp, ConeRotSOC, ConeZero, splitm, splitm_mut};
use crate::MatBuild;

//

pub struct ProbQCQPOpC<L: LinAlgEx>
{
    ph_l: PhantomData<L>,
    n: usize,
}

impl<L: LinAlgEx> Operator<L> for ProbQCQPOpC<L>
{
    fn size(&self) -> (usize, usize)
    {
        (self.n + 1, 1)
    }

    fn op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        let n = self.n;

        splitm_mut!(y, (y_n; n), (y_t; 1));

        // y_n = 0*x + b*y_n;
        L::scale(beta, &mut y_n);

        // y_t = a*1*x + b*y_t;
        L::scale(beta, &mut y_t);
        L::add(alpha, x, &mut y_t);
    }

    fn trans_op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        let n = self.n;

        splitm!(x, (_x_n; n), (x_t; 1));

        // y = 0*x_n + a*1*x_t + b*y;
        L::scale(beta, y);
        L::add(alpha, &x_t, y);
    }

    fn absadd_cols(&self, tau: &mut L::Sl)
    {
        let val_tau = tau.get(0) + L::F::one();
        tau.set(0, val_tau);
    }

    fn absadd_rows(&self, sigma: &mut L::Sl)
    {
        let n = self.n;
        let val_sigma = sigma.get(n) + L::F::one();
        sigma.set(n, val_sigma);
    }
}

//

pub struct ProbQCQPOpA<'a, L: LinAlgEx>
{
    syms_p_sqrt: Vec<MatOp<'a, L>>,
    vecs_q: Vec<MatOp<'a, L>>,
    mat_a: MatOp<'a, L>,
}

impl<'a, L: LinAlgEx> ProbQCQPOpA<'a, L>
{
    fn dim(&self) -> (usize, usize, usize)
    {
        let m1 = self.syms_p_sqrt.len();
        let (p, n) = self.mat_a.size();

        (n, m1, p)
    }
}

impl<'a, L: LinAlgEx> Operator<L> for ProbQCQPOpA<'a, L>
{
    fn size(&self) -> (usize, usize)
    {
        let (n, m1, p) = self.dim();

        (m1 * (2 + n) + p, n + 1)
    }

    fn op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        let (n, m1, p) = self.dim();

        splitm!(x, (x_n; n), (x_t; 1));

        for (i, (sym_p_sqrt, vec_q)) in self.syms_p_sqrt.iter().zip(&self.vecs_q).enumerate() {
            splitm_mut!(y, (_y_done; i * (2 + n)), (y_r; 1), (y_s; 1), (y_n; n));

            // y_r = 0*x_n + 0*x_t + b*y_r
            L::scale(beta, &mut y_r);

            // y_s = a*vec_q^T*x_n * a*-1*x_t + b*y_s  (i = 0)
            //     = a*vec_q^T*x_n *    0*x_t + b*y_s  (i > 0)
            vec_q.trans_op(alpha, &x_n, beta, &mut y_s);
            if i == 0 {
                L::add(-alpha, &x_t, &mut y_s);
            }

            // y_n = a*-sym_p_sqrt*x_n + 0*x_t + b*y_n
            sym_p_sqrt.op(-alpha, &x_n, beta, &mut y_n);
        }

        splitm_mut!(y, (_y_done; m1 * (2 + n)), (y_p; p));

        // y_p = a*mat_a*x_n + 0*x_t + b*y_p
        self.mat_a.op(alpha, &x_n, beta, &mut y_p);
    }

    fn trans_op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        let (n, m1, p) = self.dim();

        splitm_mut!(y, (y_n; n), (y_t; 1));

        let f1 = L::F::one();

        // y_n = b*y_n + ...
        // y_t = b*y_t + ...
        L::scale(beta, &mut y_n);
        L::scale(beta, &mut y_t);

        for (i, (sym_p_sqrt, vec_q)) in self.syms_p_sqrt.iter().zip(&self.vecs_q).enumerate() {
            splitm!(x, (_x_done; i * (2 + n)), (_x_r; 1), (x_s; 1), (x_n; n));

            // y_n = ... + 0*x_r * a*vec_q*x_s + a*-sym_p_sqrt*x_n + ...
            vec_q.op(alpha, &x_s, f1, &mut y_n);
            sym_p_sqrt.op(-alpha, &x_n, f1, &mut y_n);

            // y_t = ... + 0*x_r + a*-1*x_s + 0*x_n + ...  (i = 0)
            //     = ... + 0*x_r +    0*x_s + 0*x_n + ...  (i > 0)
            if i == 0 {
                L::add(-alpha, &x_s, &mut y_t);
            }
        }

        splitm!(x, (_x_done; m1 * (2 + n)), (x_p; p));

        // y_n = .. + a*mat_a^T*x_p
        self.mat_a.trans_op(alpha, &x_p, f1, &mut y_n);

        // y_t = .. + 0*x_p
    }

    fn absadd_cols(&self, tau: &mut L::Sl)
    {
        let (n, _m1, _p) = self.dim();

        splitm_mut!(tau, (tau_n; n), (tau_t; 1));

        for vec_q in &self.vecs_q {
            vec_q.absadd_rows(&mut tau_n);
        }
        for sym_p_sqrt in &self.syms_p_sqrt {
            sym_p_sqrt.absadd_cols(&mut tau_n);
        }
        self.mat_a.absadd_cols(&mut tau_n);

        let val_tau_t = tau_t.get(0) + L::F::one();
        tau_t.set(0, val_tau_t);
    }

    fn absadd_rows(&self, sigma: &mut L::Sl)
    {
        let (n, m1, p) = self.dim();

        for (i, (sym_p_sqrt, vec_q)) in self.syms_p_sqrt.iter().zip(&self.vecs_q).enumerate() {
            splitm_mut!(sigma, (_sigma_done; i * (2 + n)), (_sigma_r; 1), (sigma_s; 1), (sigma_n; n));

            vec_q.absadd_cols(&mut sigma_s);
            if i == 0 {
                let val_sigma_s = sigma_s.get(0) + L::F::one();
                sigma_s.set(0, val_sigma_s);
            }

            sym_p_sqrt.absadd_rows(&mut sigma_n);
        }

        splitm_mut!(sigma, (_sigma_done; m1 * (2 + n)), (sigma_p; p));

        self.mat_a.absadd_rows(&mut sigma_p);
    }
}

//

pub struct ProbQCQPOpB<'a, L: LinAlgEx>
{
    n: usize,
    scls_r: &'a[L::F],
    abssum_scls_r: L::F,
    vec_b: MatOp<'a, L>,
}

impl<'a, L: LinAlgEx> ProbQCQPOpB<'a, L>
{
    fn dim(&self) -> (usize, usize, usize)
    {
        let n = self.n;
        let m1 = self.scls_r.len();
        let (p, one) = self.vec_b.size();
        assert_eq!(one, 1);

        (n, m1, p)
    }
}

impl<'a, L: LinAlgEx> Operator<L> for ProbQCQPOpB<'a, L>
{
    fn size(&self) -> (usize, usize)
    {
        let (n, m1, p) = self.dim();

        (m1 * (2 + n) + p, 1)
    }

    fn op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        let (n, m1, p) = self.dim();

        for (i, scl_r) in self.scls_r.iter().enumerate() {
            splitm_mut!(y, (_y_done; i * (2 + n)), (y_r; 1), (y_s; 1), (y_n; n));

            // y_r = a*1*x + b*y_r
            L::scale(beta, &mut y_r);
            L::add(alpha, x, &mut y_r);

            // y_s = a*-scl_r*x + b*y_s
            L::scale(beta, &mut y_s);
            L::add(-alpha * *scl_r, x, &mut y_s);

            // y_n = 0*x + b*y_n
            L::scale(beta, &mut y_n);
        }

        splitm_mut!(y, (_y_done; m1 * (2 + n)), (y_p; p));

        // y_p = a*vec_b*x + b*y_p
        self.vec_b.op(alpha, x, beta, &mut y_p);
    }

    fn trans_op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        let (n, m1, p) = self.dim();

        let f1 = L::F::one();

        // y = b*y + ...
        L::scale(beta, y);

        for (i, scl_r) in self.scls_r.iter().enumerate() {
            splitm!(x, (_x_done; i * (2 + n)), (x_r; 1), (x_s; 1), (_x_n; n));

            // y = ... + a*1*x_r + a*-scl_r*x_s + 0*x_n + ...
            L::add(alpha, &x_r, y);
            L::add(-alpha * *scl_r, &x_s, y);
        }

        splitm!(x, (_x_done; m1 * (2 + n)), (x_p; p));

        // y = ... + a*vec_b^T*x_p
        self.vec_b.trans_op(alpha, &x_p, f1, y);
    }

    fn absadd_cols(&self, tau: &mut L::Sl)
    {
        let (_n, m1, _p) = self.dim();

        let val_tau = tau.get(0) + cast(m1).unwrap() + self.abssum_scls_r;
        tau.set(0, val_tau);
        self.vec_b.absadd_cols(tau);
    }

    fn absadd_rows(&self, sigma: &mut L::Sl)
    {
        let (n, m1, p) = self.dim();

        for (i, scl_r) in self.scls_r.iter().enumerate() {
            splitm_mut!(sigma, (_sigma_done; i * (2 + n)), (sigma_r; 1), (sigma_s; 1), (_sigma_n; n));

            let val_sigma_r = sigma_r.get(0) + L::F::one();
            sigma_r.set(0, val_sigma_r);
            let val_sigma_s = sigma_s.get(0) + scl_r.abs();
            sigma_s.set(0, val_sigma_s);
        }

        splitm_mut!(sigma, (_sigma_done; m1 * (2 + n)), (sigma_p; p));

        self.vec_b.absadd_rows(&mut sigma_p);
    }
}

//

pub struct ProbQCQPCone<L: LinAlgEx>
{
    n: usize,
    m1: usize,
    p: usize,
    cone_rotsoc: ConeRotSOC<L>,
    cone_zero: ConeZero<L>,
}

impl<L: LinAlgEx> Cone<L> for ProbQCQPCone<L>
{
    fn proj(&mut self, dual_cone: bool, x: &mut L::Sl) -> Result<(), ()>
    {
        let n = self.n;
        let m1 = self.m1;
        let p = self.p;

        for i in 0.. m1 {
            splitm_mut!(x, (_x_done; i * (2 + n)), (x_rsn; 2 + n));

            self.cone_rotsoc.proj(dual_cone, &mut x_rsn)?;
        }

        splitm_mut!(x, (_x_done; m1 * (2 + n)), (x_p; p));

        self.cone_zero.proj(dual_cone, &mut x_p)?;
        Ok(())
    }

    fn product_group<G: Fn(&mut L::Sl) + Copy>(&self, dp_tau: &mut L::Sl, group: G)
    {
        let n = self.n;
        let m1 = self.m1;
        let p = self.p;

        for i in 0.. m1 {
            splitm_mut!(dp_tau, (_t_done; i * (2 + n)), (t_rsn; 2 + n));

            self.cone_rotsoc.product_group(&mut t_rsn, group);
        }

        splitm_mut!(dp_tau, (_t_done; m1 * (2 + n)), (t_p; p));

        self.cone_zero.product_group(&mut t_p, group);
    }
}

//

/// Quadratically constrained quadratic program
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
/// 
/// The problem is
/// \\[
/// \begin{array}{ll}
/// {\rm minimize} & {1 \over 2} x^T P_0 x + q_0^T x + r_0 \\\\
/// {\rm subject \ to} & {1 \over 2} x^T P_i x + q_i^T x + r_i \le 0 \qquad (i = 1, \ldots, m) \\\\
/// & A x = b,
/// \end{array}
/// \\]
/// where
/// - variables \\( x \in \mathbb{R}^n \\)
/// - \\( P_j \in \mathcal{S}_{+}^n,\ q_j \in \mathbb{R}^n,\ r_j \in \mathbb{R} \\) for \\( j = 0, \ldots, m \\)
/// - \\( A \in \mathbb{R}^{p \times n},\ b \in \mathbb{R}^p \\).
/// 
/// The representation as a conic linear program is as follows:
/// \\[
/// \begin{array}{ll}
/// {\rm minimize} & t \\\\
/// {\rm subject \ to} &
///   \left[ \begin{array}{c}
///   0 & 0 \\\\
///   q_0^T & -1 \\\\
///   -P_0^{1 \over 2} & 0
///   \end{array} \right]
///   \left[ \begin{array}{c}
///   x \\\\ t
///   \end{array} \right] + s_0 =
///   \left[ \begin{array}{c}
///   1 \\\\ -r_0 \\\\ 0
///   \end{array} \right] \\\\
/// & \left[ \begin{array}{c}
///   0 \\\\ q_i^T \\\\ -P_i^{1 \over 2}
///   \end{array} \right]
///   x + s_i =
///   \left[ \begin{array}{c}
///   1 \\\\ -r_i \\\\ 0
///   \end{array} \right] \qquad (i = 1, \ldots, m) \\\\
/// & A x + s_z = b \\\\
/// & \lbrace s_0, \ldots, s_m, s_z \rbrace
///   \in \mathcal{Q}_r^{2 + n} \times \cdots \times \mathcal{Q}_r^{2 + n} \times \lbrace 0 \rbrace^p.
/// \end{array}
/// \\]
/// 
/// \\( \mathcal{Q}_r \\) is a rotated second-order (or quadratic) cone (see [`ConeRotSOC`]).
pub struct ProbQCQP<L: LinAlgEx>
{
    vecs_q: Vec<MatBuild<L>>,
    scls_r: Vec<L::F>,
    mat_a: MatBuild<L>,
    vec_b: MatBuild<L>,

    syms_p_sqrt: Vec<MatBuild<L>>,

    w_solver: Vec<L::F>,
}

impl<L: LinAlgEx> ProbQCQP<L>
{
    /// Creates a QCQP with given data.
    /// 
    /// Returns the [`ProbQCQP`] instance.
    /// * `syms_p` is \\(P_0, \\ldots, P_m\\) each of which shall belong to [`totsu_core::MatType::SymPack`].
    /// * `vecs_q` is \\(q_0, \\ldots, q_m\\).
    /// * `scls_r` is \\(r_0, \\ldots, r_m\\).
    /// * `mat_a` is \\(A\\).
    /// * `vec_b` is \\(b\\).
    /// * `eps_zero` should be the same value as [`totsu_core::solver::SolverParam::eps_zero`].
    pub fn new(
        syms_p: Vec<MatBuild<L>>, vecs_q: Vec<MatBuild<L>>, scls_r: Vec<L::F>,
        mat_a: MatBuild<L>, vec_b: MatBuild<L>,
        eps_zero: L::F) -> Self
    {
        let (p, n) = mat_a.size();
        let m1 = syms_p.len();
    
        assert_eq!(syms_p.len(), m1);
        assert_eq!(vecs_q.len(), m1);
        assert_eq!(scls_r.len(), m1);
        for i in 0.. m1 {
            assert!(syms_p[i].is_sympack());
            assert_eq!(syms_p[i].size(), (n, n));
            assert_eq!(vecs_q[i].size(), (n, 1));
        }
        assert_eq!(mat_a.size(), (p, n));
        assert_eq!(vec_b.size(), (p, 1));

        let mut syms_p_sqrt = syms_p;

        for sym_p_sqrt in &mut syms_p_sqrt {
            sym_p_sqrt.set_sqrt(eps_zero);

        }

        ProbQCQP {
            vecs_q,
            scls_r,
            mat_a,
            vec_b,
            syms_p_sqrt,
            w_solver: Vec::new(),
        }
    }
    
    /// Generates the problem data structures to be fed to [`Solver::solve`].
    /// 
    /// Returns a tuple of operators, a cone and a work slice.
    pub fn problem(&mut self) -> (ProbQCQPOpC<L>, ProbQCQPOpA<L>, ProbQCQPOpB<L>, ProbQCQPCone<L>, &mut[L::F])
    {
        let (p, n) = self.mat_a.size();
        let m1 = self.syms_p_sqrt.len();
    
        let f0 = L::F::zero();

        let op_c = ProbQCQPOpC {
            ph_l: PhantomData,
            n,
        };
        let op_a = ProbQCQPOpA {
            syms_p_sqrt: self.syms_p_sqrt.iter().map(|m| m.as_op()).collect(),
            vecs_q: self.vecs_q.iter().map(|m| m.as_op()).collect(),
            mat_a: self.mat_a.as_op(),
        };
        let op_b = ProbQCQPOpB {
            n,
            scls_r: &self.scls_r,
            abssum_scls_r: L::abssum(&L::Sl::new_ref(&self.scls_r), 1),
            vec_b: self.vec_b.as_op(),
        };

        let cone = ProbQCQPCone {
            n,
            m1,
            p,
            cone_rotsoc: ConeRotSOC::new(),
            cone_zero: ConeZero::new(),
        };

        self.w_solver.resize(Solver::<L>::query_worklen(op_a.size()), f0);

        (op_c, op_a, op_b, cone, self.w_solver.as_mut())
    }
}
