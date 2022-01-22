use std::prelude::v1::*;
use num_traits::Float;
use core::marker::PhantomData;
use crate::solver::Solver;
use crate::linalg::LinAlgEx;
use crate::operator::{Operator, MatBuild};
use crate::cone::{Cone, ConeRotSOC, ConeRPos, ConeZero};
use crate::utils::*;

//

pub struct ProbQPOpC<L, F>
where L: LinAlgEx<F>, F: Float
{
    ph_l: PhantomData<L>,
    ph_f: PhantomData<F>,
    n: usize,
}

impl<L, F> Operator<F> for ProbQPOpC<L, F>
where L: LinAlgEx<F>, F: Float
{
    fn size(&self) -> (usize, usize)
    {
        (self.n + 1, 1)
    }

    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let n = self.n;
        let (y_n, y_t) = y.split2(n, 1).unwrap();

        // y_n = 0*x + b*y_n;
        L::scale(beta, y_n);

        // y_t = a*1*x + b*y_t;
        L::scale(beta, y_t);
        L::add(alpha, x, y_t);
    }

    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let n = self.n;
        let (_x_n, x_t) = x.split2(n, 1).unwrap();

        // y = 0*x_n + a*1*x_t + b*y;
        L::scale(beta, y);
        L::add(alpha, x_t, y);
    }

    fn abssum_cols(&self, beta: F, tau: &mut[F])
    {
        crate::operator::reffn::abssum_cols::<L, _, _>(
            self.size(),
            |x, y| self.op(F::one(), x, F::zero(), y),
            beta, tau
        );
    }

    fn abssum_rows(&self, beta: F, sigma: &mut[F])
    {
        crate::operator::reffn::abssum_rows::<L, _, _>(
            self.size(),
            |x, y| self.trans_op(F::one(), x, F::zero(), y),
            beta, sigma
        );
    }
}

//

pub struct ProbQPOpA<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    sym_p_sqrt: &'a MatBuild<L, F>,
    vec_q: &'a MatBuild<L, F>,
    mat_g: &'a MatBuild<L, F>,
    mat_a: &'a MatBuild<L, F>,
}

impl<'a, L, F> ProbQPOpA<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn dim(&self) -> (usize, usize, usize)
    {
        let (n, n_) = self.sym_p_sqrt.size();
        assert_eq!(n, n_);
        let (m, n_) = self.mat_g.size();
        assert_eq!(n, n_);
        let (p, n_) = self.mat_a.size();
        assert_eq!(n, n_);

        (n, m, p)
    }
}

impl<'a, L, F> Operator<F> for ProbQPOpA<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn size(&self) -> (usize, usize)
    {
        let (n, m, p) = self.dim();

        ((2 + n) + m + p, n + 1)
    }

    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (n, m, p) = self.dim();
        let (x_n, x_t) = x.split2(n, 1).unwrap();
        let (y_r, y_s, y_n, y_m, y_p) = y.split5(1, 1, n, m, p).unwrap();

        // y_r = 0*x_n + 0*x_t + b*y_r
        L::scale(beta, y_r);

        // y_s = a*vec_q^T*x_n * a*-1*x_t + b*y_s
        self.vec_q.trans_op(alpha, x_n, beta, y_s);
        L::add(-alpha, x_t, y_s);

        // y_n = a*-sym_p_sqrt*x_n + 0*x_t + b*y_n
        self.sym_p_sqrt.op(-alpha, x_n, beta, y_n);

        // y_m = a*mat_g*x_n + 0*x_t + b*y_m
        self.mat_g.op(alpha, x_n, beta, y_m);

        // y_p = a*mat_a*x_n + 0*x_t + b*y_p
        self.mat_a.op(alpha, x_n, beta, y_p);
    }

    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (n, m, p) = self.dim();
        let (_x_r, x_s, x_n, x_m, x_p) = x.split5(1, 1, n, m, p).unwrap();
        let (y_n, y_t) = y.split2(n, 1).unwrap();

        let f1 = F::one();
        
        // y_n = 0*x_r * a*vec_q*x_s + a*-sym_p_sqrt*x_n + a*mat_g^T*x_m + a*mat_a^T*x_p + b*y_n
        self.vec_q.op(alpha, x_s, beta, y_n);
        self.sym_p_sqrt.op(-alpha, x_n, f1, y_n);
        self.mat_g.trans_op(alpha, x_m, f1, y_n);
        self.mat_a.trans_op(alpha, x_p, f1, y_n);

        // y_t = 0*x_r + a*-1*x_s + 0*x_n + 0*x_m + 0*x_p + b*y_t
        L::scale(beta, y_t);
        L::add(-alpha, x_s, y_t);
    }

    fn abssum_cols(&self, beta: F, tau: &mut[F])
    {
        crate::operator::reffn::abssum_cols::<L, _, _>(
            self.size(),
            |x, y| self.op(F::one(), x, F::zero(), y),
            beta, tau
        );
    }

    fn abssum_rows(&self, beta: F, sigma: &mut[F])
    {
        crate::operator::reffn::abssum_rows::<L, _, _>(
            self.size(),
            |x, y| self.trans_op(F::one(), x, F::zero(), y),
            beta, sigma
        );
    }
}

//

pub struct ProbQPOpB<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    n: usize,
    vec_h: &'a MatBuild<L, F>,
    vec_b: &'a MatBuild<L, F>,
}

impl<'a, L, F> ProbQPOpB<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn dim(&self) -> (usize, usize, usize)
    {
        let (m, one) = self.vec_h.size();
        assert_eq!(one, 1);
        let (p, one) = self.vec_b.size();
        assert_eq!(one, 1);

        (self.n, m, p)
    }
}

impl<'a, L, F> Operator<F> for ProbQPOpB<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn size(&self) -> (usize, usize)
    {
        let (n, m, p) = self.dim();

        ((2 + n) + m + p, 1)
    }

    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (n, m, p) = self.dim();
        let (y_r, y_sn, y_m, y_p) = y.split4(1, 1 + n, m, p).unwrap();

        // y_r = a*1*x + b*y_r
        L::scale(beta, y_r);
        L::add(alpha, x, y_r);

        // y_sn = 0*x + b*y_sn
        L::scale(beta, y_sn);

        // y_m = a*vec_h*x + b*y_m
        self.vec_h.op(alpha, x, beta, y_m);

        // y_p = a*vec_b*x + b*y_p
        self.vec_b.op(alpha, x, beta, y_p);
    }

    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (n, m, p) = self.dim();
        let (x_r, _x_sn, x_m, x_p) = x.split4(1, 1 + n, m, p).unwrap();

        let f1 = F::one();

        // y = a*1*x_r + 0*x_sn + a*vec_h^T*x_m + a*vec_b^T*x_p + b*y
        self.vec_h.trans_op(alpha, x_m, beta, y);
        self.vec_b.trans_op(alpha, x_p, f1, y);
        L::add(alpha, x_r, y);
    }

    fn abssum_cols(&self, beta: F, tau: &mut[F])
    {
        crate::operator::reffn::abssum_cols::<L, _, _>(
            self.size(),
            |x, y| self.op(F::one(), x, F::zero(), y),
            beta, tau
        );
    }

    fn abssum_rows(&self, beta: F, sigma: &mut[F])
    {
        crate::operator::reffn::abssum_rows::<L, _, _>(
            self.size(),
            |x, y| self.trans_op(F::one(), x, F::zero(), y),
            beta, sigma
        );
    }
}

//

pub struct ProbQPCone<L, F>
where L: LinAlgEx<F>, F: Float
{
    n: usize,
    m: usize,
    p: usize,
    cone_rotsoc: ConeRotSOC<L, F>,
    cone_rpos: ConeRPos<F>,
    cone_zero: ConeZero<F>,
}

impl<'a, L, F> Cone<F> for ProbQPCone<L, F>
where L: LinAlgEx<F>, F: Float
{
    fn proj(&mut self, dual_cone: bool, x: &mut[F]) -> Result<(), ()>
    {
        let (n, m, p) = (self.n, self.m, self.p);
        let (x_rsn, x_m, x_p) = x.split3(2 + n, m, p).unwrap();

        self.cone_rotsoc.proj(dual_cone, x_rsn)?;
        self.cone_rpos.proj(dual_cone, x_m)?;
        self.cone_zero.proj(dual_cone, x_p)?;
        Ok(())
    }

    fn product_group<G: Fn(&mut[F]) + Copy>(&self, dp_tau: &mut[F], group: G)
    {
        let (n, m, p) = (self.n, self.m, self.p);
        let (t_rsn, t_m, t_p) = dp_tau.split3(2 + n, m, p).unwrap();

        self.cone_rotsoc.product_group(t_rsn, group);
        self.cone_rpos.product_group(t_m, group);
        self.cone_zero.product_group(t_p, group);
    }

}

//

/// Quadratic program
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
/// 
/// The problem is
/// \\[
/// \begin{array}{ll}
/// {\rm minimize} & {1 \over 2} x^T P x + q^T x + r \\\\
/// {\rm subject \ to} & G x \preceq h \\\\
/// & A x = b,
/// \end{array}
/// \\]
/// where
/// - variables \\( x \in \mathbb{R}^n \\)
/// - \\( P \in \mathcal{S}_{+}^n,\ q \in \mathbb{R}^n,\ r \in \mathbb{R} \\)
/// - \\( G \in \mathbb{R}^{m \times n},\ h \in \mathbb{R}^m \\)
/// - \\( A \in \mathbb{R}^{p \times n},\ b \in \mathbb{R}^p \\).
/// 
/// In the following, \\( r \\) does not appear since it does not matter.
/// 
/// The representation as a conic linear program is as follows:
/// \\[
/// \begin{array}{ll}
/// {\rm minimize} & t \\\\
/// {\rm subject \ to} &
///   \left[ \begin{array}{ccc}
///   0 & 0 \\\\
///   q^T & -1 \\\\
///   -P^{1 \over 2} & 0 \\\\
///   G & 0 \\\\
///   A & 0
///   \end{array} \right]
///   \left[ \begin{array}{c}
///   x \\\\ t
///   \end{array} \right]
///   + s =
///   \left[ \begin{array}{c}
///   1 \\\\ 0 \\\\ 0 \\\\ h \\\\ b
///   \end{array} \right] \\\\
/// & s \in \mathcal{Q}_r^{2 + n} \times \mathbb{R}\_+^m \times \lbrace 0 \rbrace^p.
/// \end{array}
/// \\]
/// 
/// \\( \mathcal{Q}_r \\) is a rotated second-order (or quadratic) cone (see [`ConeRotSOC`]).
pub struct ProbQP<L, F>
where L: LinAlgEx<F>, F: Float
{
    vec_q: MatBuild<L, F>,
    mat_g: MatBuild<L, F>,
    vec_h: MatBuild<L, F>,
    mat_a: MatBuild<L, F>,
    vec_b: MatBuild<L, F>,

    sym_p_sqrt: MatBuild<L, F>,

    w_solver: Vec<F>,
}

impl<L, F> ProbQP<L, F>
where L: LinAlgEx<F>, F: Float
{
    /// Creates a QP with given data.
    /// 
    /// Returns a [`ProbQP`] instance.
    /// * `sym_p` is \\(P\\) which shall belong to [`crate::operator::MatType::SymPack`].
    /// * `vec_q` is \\(q\\).
    /// * `mat_g` is \\(G\\).
    /// * `vec_h` is \\(h\\).
    /// * `mat_a` is \\(A\\).
    /// * `vec_b` is \\(b\\).
    /// * `eps_zero` should be the same value as [`crate::solver::SolverParam::eps_zero`].
    pub fn new(
        sym_p: MatBuild<L, F>, vec_q: MatBuild<L, F>,
        mat_g: MatBuild<L, F>, vec_h: MatBuild<L, F>,
        mat_a: MatBuild<L, F>, vec_b: MatBuild<L, F>,
        eps_zero: F) -> Self
    {
        let n = vec_q.size().0;
        let m = vec_h.size().0;
        let p = vec_b.size().0;

        assert!(sym_p.is_sympack());
        assert_eq!(sym_p.size(), (n, n));
        assert_eq!(vec_q.size(), (n, 1));
        assert_eq!(mat_g.size(), (m, n));
        assert_eq!(vec_h.size(), (m, 1));
        assert_eq!(mat_a.size(), (p, n));
        assert_eq!(vec_b.size(), (p, 1));

        let sym_p_sqrt = sym_p.sqrt(eps_zero);

        ProbQP {
            vec_q,
            mat_g,
            vec_h,
            mat_a,
            vec_b,
            sym_p_sqrt,
            w_solver: Vec::new(),
        }
    }

    /// Generates the problem data structures to be fed to [`crate::solver::Solver::solve`].
    /// 
    /// Returns a tuple of operators, a cone and a work slice.
    pub fn problem(&mut self) -> (ProbQPOpC<L, F>, ProbQPOpA<L, F>, ProbQPOpB<L, F>, ProbQPCone<L, F>, &mut[F])
    {
        let n = self.vec_q.size().0;
        let m = self.vec_h.size().0;
        let p = self.vec_b.size().0;

        let f0 = F::zero();

        let op_c = ProbQPOpC {
            ph_l: PhantomData,
            ph_f: PhantomData,
            n,
        };
        let op_a = ProbQPOpA {
            sym_p_sqrt: &self.sym_p_sqrt,
            vec_q: &self.vec_q,
            mat_g: &self.mat_g,
            mat_a: &self.mat_a,
        };
        let op_b = ProbQPOpB {
            n,
            vec_h: &self.vec_h,
            vec_b: &self.vec_b,
        };

        let cone = ProbQPCone {
            n, m, p,
            cone_rotsoc: ConeRotSOC::new(),
            cone_rpos: ConeRPos::new(),
            cone_zero: ConeZero::new(),
        };

        self.w_solver.resize(Solver::<L, _>::query_worklen(op_a.size()), f0);

        (op_c, op_a, op_b, cone, self.w_solver.as_mut())
    }
}
