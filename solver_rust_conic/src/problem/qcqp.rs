use num::Float;
use core::marker::PhantomData;
use crate::solver::Solver;
use crate::linalg::LinAlgEx;
use crate::operator::{Operator, MatBuild};
use crate::cone::{Cone, ConeRotSOC, ConeZero};
use crate::utils::*;

//

pub struct ProbQCQPOpC<L, F>
where L: LinAlgEx<F>, F: Float
{
    ph_l: PhantomData<L>,
    ph_f: PhantomData<F>,
    n: usize,
}

impl<L, F> Operator<F> for ProbQCQPOpC<L, F>
where L: LinAlgEx<F>, F: Float
{
    fn size(&self) -> (usize, usize)
    {
        let n = self.n;

        (n + 2, 1)
    }

    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let n = self.n;
        let (y_n, y_t, y_1) = y.split3(n, 1, 1).unwrap();

        // y_n = 0*x + b*y_n;
        L::scale(beta, y_n);

        // y_t = a*1*x + b*y_t;
        L::scale(beta, y_t);
        L::add(alpha, x, y_t);

        // y_1 = a*0*x + b*y_1;
        L::scale(beta, y_1);
    }

    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let n = self.n;
        let (_x_n, x_t, _x_1) = x.split3(n, 1, 1).unwrap();

        // y = 0*x_n + a*1*x_t + 0*x_1 + b*y;
        L::scale(beta, y);
        L::add(alpha, x_t, y);
    }
}

//

pub struct ProbQCQPOpA<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    syms_p_sqrt: &'a[MatBuild<L, F>],
    vecs_q: &'a[MatBuild<L, F>],
    mat_a: &'a MatBuild<L, F>,
}

impl<'a, L, F> ProbQCQPOpA<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn dim(&self) -> (usize, usize, usize)
    {
        let m1 = self.syms_p_sqrt.len();
        let (p, n) = self.mat_a.size();

        (n, m1, p)
    }
}

impl<'a, L, F> Operator<F> for ProbQCQPOpA<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn size(&self) -> (usize, usize)
    {
        let (n, m1, p) = self.dim();

        (m1 * (2 + n) + p + 1, n + 2)
    }

    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (n, _m1, p) = self.dim();
        let (x_n, x_t, x_1) = x.split3(n, 1, 1).unwrap();

        let mut i = 0;
        let mut spl_y = y;
        for (sym_p_sqrt, vec_q) in self.syms_p_sqrt.iter().zip(self.vecs_q) {
            let (y_r, spl) = spl_y.split_at_mut(1);
            let (y_s, spl) = spl.split_at_mut(1);
            let (y_n, spl) = spl.split_at_mut(n);
            spl_y = spl;

            // y_r = 0*x_n + 0*x_t + a*-1*x_1 + b*y_r
            L::scale(beta, y_r);
            L::add(-alpha, x_1, y_r);

            // y_s = a*vec_q^T*x_n * a*-1*x_t + 0*x_1 + b*y_s  (i = 0)
            //     = a*vec_q^T*x_n *    0*x_t + 0*x_1 + b*y_s  (i > 0)
            vec_q.trans_op(alpha, x_n, beta, y_s);
            if i == 0 {
                L::add(-alpha, x_t, y_s);
            }

            // y_n = a*-sym_p_sqrt*x_n + 0*x_t + 0*x_1 + b*y_n
            sym_p_sqrt.op(-alpha, x_n, beta, y_n);

            i += 1;
        }

        let (y_p, y_1) = spl_y.split2(p, 1).unwrap();

        // y_p = a*mat_a*x_n + 0*x_t + 0*x_1 + b*y_p
        self.mat_a.op(alpha, x_n, beta, y_p);

        // y_1 = 0*x_n + 0*x_t + a*1*x_1 + b*y_1
        L::scale(beta, y_1);
        L::add(alpha, x_1, y_1);
    }

    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (n, _m1, p) = self.dim();
        let (y_n, y_t, y_1) = y.split3(n, 1, 1).unwrap();

        let f1 = F::one();

        // y_n = b*y_n + ...
        // y_t = b*y_t + ...
        // y_1 = b*y_1 + ...
        L::scale(beta, y_n);
        L::scale(beta, y_t);
        L::scale(beta, y_1);

        let mut i = 0;
        let mut spl_x = x;
        for (sym_p_sqrt, vec_q) in self.syms_p_sqrt.iter().zip(self.vecs_q) {
            let (x_r, spl) = spl_x.split_at(1);
            let (x_s, spl) = spl.split_at(1);
            let (x_n, spl) = spl.split_at(n);
            spl_x = spl;

            // y_n = ... + 0*x_r * a*vec_q*x_s + a*-sym_p_sqrt*x_n + ...
            vec_q.op(alpha, x_s, f1, y_n);
            sym_p_sqrt.op(-alpha, x_n, f1, y_n);

            // y_t = ... + 0*x_r + a*-1*x_s + 0*x_n + ...  (i = 0)
            //     = ... + 0*x_r +    0*x_s + 0*x_n + ...  (i > 0)
            if i == 0 {
                L::add(-alpha, x_s, y_t);
            }

            // y_1 = ... + a*-1*x_r + 0*x_s + 0*x_n + ...
            L::add(-alpha, x_r, y_1);

            i += 1;
        }

        let (x_p, x_1) = spl_x.split2(p, 1).unwrap();

        // y_n = .. + a*mat_a^T*x_p + 0*x_1
        self.mat_a.trans_op(alpha, x_p, f1, y_n);

        // y_t = .. + 0*x_p + 0*x_1

        // y_1 = .. + 0*x_p + a*1*x_1
        L::add(alpha, x_1, y_1);
    }
}

//

pub struct ProbQCQPOpB<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    n: usize,
    scls_r: &'a[F],
    vec_b: &'a MatBuild<L, F>,
}

impl<'a, L, F> ProbQCQPOpB<'a, L, F>
where L: LinAlgEx<F>, F: Float
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

impl<'a, L, F> Operator<F> for ProbQCQPOpB<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn size(&self) -> (usize, usize)
    {
        let (n, m1, p) = self.dim();

        (m1 * (2 + n) + p + 1, 1)
    }

    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (n, _m1, p) = self.dim();

        let mut spl_y = y;
        for scl_r in self.scls_r {
            let (y_r, spl) = spl_y.split_at_mut(1);
            let (y_s, spl) = spl.split_at_mut(1);
            let (y_n, spl) = spl.split_at_mut(n);
            spl_y = spl;

            // y_r = 0*x + b*y_r
            L::scale(beta, y_r);

            // y_s = a*-scl_r*x + b*y_s
            L::scale(beta, y_s);
            L::add(-alpha * *scl_r, x, y_s);

            // y_n = 0*x + b*y_n
            L::scale(beta, y_n);
        }

        let (y_p, y_1) = spl_y.split2(p, 1).unwrap();

        // y_p = a*vec_b*x + b*y_p
        self.vec_b.op(alpha, x, beta, y_p);

        // y_1 = a*1*x + b*y_1
        L::scale(beta, y_1);
        L::add(alpha, x, y_1);
    }

    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (n, _m1, p) = self.dim();

        let f1 = F::one();

        // y = b*y + ...
        L::scale(beta, y);

        let mut spl_x = x;
        for scl_r in self.scls_r {
            let (_x_r, spl) = spl_x.split_at(1);
            let (x_s, spl) = spl.split_at(1);
            let (_x_n, spl) = spl.split_at(n);
            spl_x = spl;

            // y = ... + 0*x_r + a*-scl_r*x_s + 0*x_n + ...
            L::add(-alpha * *scl_r, x_s, y);
        }

        let (x_p, x_1) = spl_x.split2(p, 1).unwrap();

        // y = ... + a*vec_b^T*x_p * a*1*x_1
        self.vec_b.trans_op(alpha, x_p, f1, y);
        L::add(alpha, x_1, y);
    }
}

//

pub struct ProbQCQPCone<L, F>
where L: LinAlgEx<F>, F: Float
{
    n: usize,
    m1: usize,
    cone_rotsoc: ConeRotSOC<L, F>,
    cone_zero: ConeZero<F>,
}

impl<L, F> Cone<F> for ProbQCQPCone<L, F>
where L: LinAlgEx<F>, F: Float
{
    fn proj(&mut self, dual_cone: bool, eps_zero: F, x: &mut[F]) -> Result<(), ()>
    {
        let n = self.n;
        let m1 = self.m1;

        let mut spl_x = x;
        for _ in 0.. m1 {
            let (x_rsn, spl) = spl_x.split_at_mut(2 + n);
            spl_x = spl;

            self.cone_rotsoc.proj(dual_cone, eps_zero, x_rsn)?;
        }

        let x_p1 = spl_x;

        self.cone_zero.proj(dual_cone, eps_zero, x_p1)?;
        Ok(())
    }

    fn product_group<G: Fn(&mut[F]) + Copy>(&self, dp_tau: &mut[F], group: G)
    {
        let n = self.n;
        let m1 = self.m1;

        let mut spl_t = dp_tau;
        for _ in 0.. m1 {
            let (t_rsn, spl) = spl_t.split_at_mut(2 + n);
            spl_t = spl;

            self.cone_rotsoc.product_group(t_rsn, group);
        }

        let t_p1 = spl_t;

        self.cone_zero.product_group(t_p1, group);
    }
}

//

/// Quadratically constrained quadratic program
/// 
/// <script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML' async></script>
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
/// - variables \\( x \in {\bf R}^n \\)
/// - \\( P_j \in {\bf S}_{+}^n,\ q_j \in {\bf R}^n,\ r_j \in {\bf R} \\) for \\( j = 0, \ldots, m \\)
/// - \\( A \in {\bf R}^{p \times n},\ b \in {\bf R}^p \\).
/// 
/// The representation as a conic linear program is as follows:
/// \\[
/// \begin{array}{ll}
/// {\rm minimize} & t \\\\
/// {\rm subject \ to} &
///   \left[ \begin{array}{ccc}
///   0 & 0 & -1 \\\\
///   q_0^T & -1 & 0 \\\\
///   -P_0^{1 \over 2} & 0 & 0
///   \end{array} \right]
///   \left[ \begin{array}{c}
///   x \\\\ t \\\\ e
///   \end{array} \right]
///   + s_0 =
///   \left[ \begin{array}{c}
///   0 \\\\ -r_0 \\\\ 0
///   \end{array} \right] \\\\
/// & \left[ \begin{array}{ccc}
///   0 & 0 & -1 \\\\
///   q_i^T & 0 & 0 \\\\
///   -P_i^{1 \over 2} & 0 & 0
///   \end{array} \right]
///   \left[ \begin{array}{c}
///   x \\\\ t \\\\ e
///   \end{array} \right]
///   + s_i =
///   \left[ \begin{array}{c}
///   0 \\\\ -r_i \\\\ 0
///   \end{array} \right] \qquad (i = 1, \ldots, m) \\\\
/// & \left[ \begin{array}{ccc}
///   A & 0 & 0 \\\\
///   0 & 0 & 1
///   \end{array} \right]
///   \left[ \begin{array}{c}
///   x \\\\ t \\\\ e
///   \end{array} \right]
///   + s_z =
///   \left[ \begin{array}{c}
///   b \\\ 1
///   \end{array} \right] \\\\
/// & \lbrace s_0, \ldots, s_m, s_z \rbrace
///   \in \mathcal{Q}_r^{2 + n} \times \cdots \times \mathcal{Q}_r^{2 + n} \times \lbrace 0 \rbrace^{p + 1}.
/// \end{array}
/// \\]
/// 
/// \\( \mathcal{Q}_r \\) is a rotated second-order cone (see [`ConeRotSOC`]).
pub struct ProbQCQP<L, F>
where L: LinAlgEx<F>, F: Float
{
    vecs_q: Vec<MatBuild<L, F>>,
    scls_r: Vec<F>,
    mat_a: MatBuild<L, F>,
    vec_b: MatBuild<L, F>,

    syms_p_sqrt: Vec<MatBuild<L, F>>,

    w_solver: Vec<F>,
}

impl<L, F> ProbQCQP<L, F>
where L: LinAlgEx<F>, F: Float
{
    /// Creates a QCQP with given data.
    /// 
    /// Returns a [`ProbQCQP`] instance.
    /// * `syms_p` is \\(P_0, \\ldots, P_m\\) each of which shall belong to [`crate::operator::MatType::SymPack`].
    /// * `vecs_q` is \\(q_0, \\ldots, q_m\\).
    /// * `scls_r` is \\(r_0, \\ldots, r_m\\).
    /// * `mat_a` is \\(A\\).
    /// * `vec_b` is \\(b\\).
    /// * `eps_zero` should be the same value as [`crate::solver::SolverParam::eps_zero`].
    pub fn new(
        syms_p: Vec<MatBuild<L, F>>, vecs_q: Vec<MatBuild<L, F>>, scls_r: Vec<F>,
        mat_a: MatBuild<L, F>, vec_b: MatBuild<L, F>,
        eps_zero: F) -> Self
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
    
    /// Generates the problem data structures to be fed to [`crate::solver::Solver::solve`].
    /// 
    /// Returns a tuple of operators, a cone and a work slice.
    pub fn problem(&mut self) -> (ProbQCQPOpC<L, F>, ProbQCQPOpA<L, F>, ProbQCQPOpB<L, F>, ProbQCQPCone<L, F>, &mut[F])
    {
        let n = self.mat_a.size().1;
        let m1 = self.syms_p_sqrt.len();
    
        let f0 = F::zero();

        let op_c = ProbQCQPOpC {
            ph_l: PhantomData,
            ph_f: PhantomData,
            n,
        };
        let op_a = ProbQCQPOpA {
            syms_p_sqrt: &self.syms_p_sqrt,
            vecs_q: &self.vecs_q,
            mat_a: &self.mat_a,
        };
        let op_b = ProbQCQPOpB {
            n,
            scls_r: &self.scls_r,
            vec_b: &self.vec_b,
        };

        let cone = ProbQCQPCone {
            n,
            m1,
            cone_rotsoc: ConeRotSOC::new(),
            cone_zero: ConeZero::new(),
        };

        self.w_solver.resize(Solver::<L, _>::query_worklen(op_a.size()), f0);

        (op_c, op_a, op_b, cone, self.w_solver.as_mut())
    }
}
