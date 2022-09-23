use std::marker::PhantomData;
use num_traits::{Zero, One};
use totsu_core::solver::{Solver, SliceLike, Operator, Cone};
use totsu_core::{LinAlgEx, MatOp, ConeRotSOC, ConeRPos, ConeZero, splitm, splitm_mut};
use crate::MatBuild;

//

pub struct ProbQPOpC<L: LinAlgEx>
{
    ph_l: PhantomData<L>,
    n: usize,
}

impl<L: LinAlgEx> Operator<L> for ProbQPOpC<L>
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

        splitm_mut!(sigma, (_sigma_n; n), (sigma_t; 1));

        let val_sigma_t = sigma_t.get(0) + L::F::one();
        sigma_t.set(0, val_sigma_t);
    }
}

//

pub struct ProbQPOpA<'a, L: LinAlgEx>
{
    sym_p_sqrt: MatOp<'a, L>,
    vec_q: MatOp<'a, L>,
    mat_g: MatOp<'a, L>,
    mat_a: MatOp<'a, L>,
}

impl<'a, L: LinAlgEx> ProbQPOpA<'a, L>
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

impl<'a, L: LinAlgEx> Operator<L> for ProbQPOpA<'a, L>
{
    fn size(&self) -> (usize, usize)
    {
        let (n, m, p) = self.dim();

        ((2 + n) + m + p, n + 1)
    }

    fn op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        let (n, m, p) = self.dim();

        splitm!(x, (x_n; n), (x_t; 1));
        splitm_mut!(y, (y_r; 1), (y_s; 1), (y_n; n), (y_m; m), (y_p; p));

        // y_r = 0*x_n + 0*x_t + b*y_r
        L::scale(beta, &mut y_r);

        // y_s = a*vec_q^T*x_n * a*-1*x_t + b*y_s
        self.vec_q.trans_op(alpha, &x_n, beta, &mut y_s);
        L::add(-alpha, &x_t, &mut y_s);

        // y_n = a*-sym_p_sqrt*x_n + 0*x_t + b*y_n
        self.sym_p_sqrt.op(-alpha, &x_n, beta, &mut y_n);

        // y_m = a*mat_g*x_n + 0*x_t + b*y_m
        self.mat_g.op(alpha, &x_n, beta, &mut y_m);

        // y_p = a*mat_a*x_n + 0*x_t + b*y_p
        self.mat_a.op(alpha, &x_n, beta, &mut y_p);
    }

    fn trans_op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        let (n, m, p) = self.dim();

        splitm!(x, (_x_r; 1), (x_s; 1), (x_n; n), (x_m; m), (x_p; p));
        splitm_mut!(y, (y_n; n), (y_t; 1));

        let f1 = L::F::one();
        
        // y_n = 0*x_r * a*vec_q*x_s + a*-sym_p_sqrt*x_n + a*mat_g^T*x_m + a*mat_a^T*x_p + b*y_n
        self.vec_q.op(alpha, &x_s, beta, &mut y_n);
        self.sym_p_sqrt.op(-alpha, &x_n, f1, &mut y_n);
        self.mat_g.trans_op(alpha, &x_m, f1, &mut y_n);
        self.mat_a.trans_op(alpha, &x_p, f1, &mut y_n);

        // y_t = 0*x_r + a*-1*x_s + 0*x_n + 0*x_m + 0*x_p + b*y_t
        L::scale(beta, &mut y_t);
        L::add(-alpha, &x_s, &mut y_t);
    }

    fn absadd_cols(&self, tau: &mut L::Sl)
    {
        let (n, _m, _p) = self.dim();

        splitm_mut!(tau, (tau_n; n), (tau_t; 1));

        self.vec_q.absadd_rows(&mut tau_n);
        self.sym_p_sqrt.absadd_cols(&mut tau_n);
        self.mat_g.absadd_cols(&mut tau_n);
        self.mat_a.absadd_cols(&mut tau_n);

        let val_tau_t = tau_t.get(0) + L::F::one();
        tau_t.set(0, val_tau_t);
    }

    fn absadd_rows(&self, sigma: &mut L::Sl)
    {
        let (n, m, p) = self.dim();

        splitm_mut!(sigma, (_sigma_r; 1), (sigma_s; 1), (sigma_n; n), (sigma_m; m), (sigma_p; p));

        self.vec_q.absadd_cols(&mut sigma_s);
        let val_sigma_s = sigma_s.get(0) + L::F::one();
        sigma_s.set(0, val_sigma_s);
        self.sym_p_sqrt.absadd_rows(&mut sigma_n);
        self.mat_g.absadd_rows(&mut sigma_m);
        self.mat_a.absadd_rows(&mut sigma_p);
    }
}

//

pub struct ProbQPOpB<'a, L: LinAlgEx>
{
    n: usize,
    vec_h: MatOp<'a, L>,
    vec_b: MatOp<'a, L>,
}

impl<'a, L: LinAlgEx> ProbQPOpB<'a, L>
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

impl<'a, L: LinAlgEx> Operator<L> for ProbQPOpB<'a, L>
{
    fn size(&self) -> (usize, usize)
    {
        let (n, m, p) = self.dim();

        ((2 + n) + m + p, 1)
    }

    fn op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        let (n, m, p) = self.dim();

        splitm_mut!(y, (y_r; 1), (y_sn; 1 + n), (y_m; m), (y_p; p));

        // y_r = a*1*x + b*y_r
        L::scale(beta, &mut y_r);
        L::add(alpha, x, &mut y_r);

        // y_sn = 0*x + b*y_sn
        L::scale(beta, &mut y_sn);

        // y_m = a*vec_h*x + b*y_m
        self.vec_h.op(alpha, x, beta, &mut y_m);

        // y_p = a*vec_b*x + b*y_p
        self.vec_b.op(alpha, x, beta, &mut y_p);
    }

    fn trans_op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        let (n, m, p) = self.dim();

        splitm!(x, (x_r; 1), (_x_sn; 1 + n), (x_m; m), (x_p; p));

        let f1 = L::F::one();

        // y = a*1*x_r + 0*x_sn + a*vec_h^T*x_m + a*vec_b^T*x_p + b*y
        self.vec_h.trans_op(alpha, &x_m, beta, y);
        self.vec_b.trans_op(alpha, &x_p, f1, y);
        L::add(alpha, &x_r, y);
    }

    fn absadd_cols(&self, tau: &mut L::Sl)
    {
        let val_tau = tau.get(0) + L::F::one();
        tau.set(0, val_tau);
        self.vec_h.absadd_cols(tau);
        self.vec_b.absadd_cols(tau);
    }

    fn absadd_rows(&self, sigma: &mut L::Sl)
    {
        let (n, m, p) = self.dim();

        splitm_mut!(sigma, (sigma_r; 1), (_sigma_sn; 1 + n), (sigma_m; m), (sigma_p; p));

        let val_sigma_r = sigma_r.get(0) + L::F::one();
        sigma_r.set(0, val_sigma_r);
        self.vec_h.absadd_rows(&mut sigma_m);
        self.vec_b.absadd_rows(&mut sigma_p);
    }
}

//

pub struct ProbQPCone<L: LinAlgEx>
{
    n: usize,
    m: usize,
    p: usize,
    cone_rotsoc: ConeRotSOC<L>,
    cone_rpos: ConeRPos<L>,
    cone_zero: ConeZero<L>,
}

impl<'a, L: LinAlgEx> Cone<L> for ProbQPCone<L>
{
    fn proj(&mut self, dual_cone: bool, x: &mut L::Sl) -> Result<(), ()>
    {
        let (n, m, p) = (self.n, self.m, self.p);

        splitm_mut!(x, (x_rsn; 2 + n), (x_m; m), (x_p; p));

        self.cone_rotsoc.proj(dual_cone, &mut x_rsn)?;
        self.cone_rpos.proj(dual_cone, &mut x_m)?;
        self.cone_zero.proj(dual_cone, &mut x_p)?;
        Ok(())
    }

    fn product_group<G: Fn(&mut L::Sl) + Copy>(&self, dp_tau: &mut L::Sl, group: G)
    {
        let (n, m, p) = (self.n, self.m, self.p);

        splitm_mut!(dp_tau, (t_rsn; 2 + n), (t_m; m), (t_p; p));

        self.cone_rotsoc.product_group(&mut t_rsn, group);
        self.cone_rpos.product_group(&mut t_m, group);
        self.cone_zero.product_group(&mut t_p, group);
    }

}

//

/// Quadratic program
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
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
pub struct ProbQP<L: LinAlgEx>
{
    vec_q: MatBuild<L>,
    mat_g: MatBuild<L>,
    vec_h: MatBuild<L>,
    mat_a: MatBuild<L>,
    vec_b: MatBuild<L>,

    sym_p_sqrt: MatBuild<L>,

    w_solver: Vec<L::F>,
}

impl<L: LinAlgEx> ProbQP<L>
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
        sym_p: MatBuild<L>, vec_q: MatBuild<L>,
        mat_g: MatBuild<L>, vec_h: MatBuild<L>,
        mat_a: MatBuild<L>, vec_b: MatBuild<L>,
        eps_zero: L::F) -> Self
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
    pub fn problem(&mut self) -> (ProbQPOpC<L>, ProbQPOpA<L>, ProbQPOpB<L>, ProbQPCone<L>, &mut[L::F])
    {
        let n = self.vec_q.size().0;
        let m = self.vec_h.size().0;
        let p = self.vec_b.size().0;

        let f0 = L::F::zero();

        let op_c = ProbQPOpC {
            ph_l: PhantomData,
            n,
        };
        let op_a = ProbQPOpA {
            sym_p_sqrt: self.sym_p_sqrt.as_op(),
            vec_q: self.vec_q.as_op(),
            mat_g: self.mat_g.as_op(),
            mat_a: self.mat_a.as_op(),
        };
        let op_b = ProbQPOpB {
            n,
            vec_h: self.vec_h.as_op(),
            vec_b: self.vec_b.as_op(),
        };

        let cone = ProbQPCone {
            n, m, p,
            cone_rotsoc: ConeRotSOC::new(),
            cone_rpos: ConeRPos::new(),
            cone_zero: ConeZero::new(),
        };

        self.w_solver.resize(Solver::<L>::query_worklen(op_a.size()), f0);

        (op_c, op_a, op_b, cone, self.w_solver.as_mut())
    }
}
