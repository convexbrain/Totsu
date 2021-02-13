use std::prelude::v1::*;
use num_traits::Float;
use crate::solver::Solver;
use crate::linalg::LinAlgEx;
use crate::operator::{Operator, MatType, MatBuild};
use crate::cone::{Cone, ConePSD, ConeZero};
use crate::utils::*;

//

pub struct ProbSDPOpC<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    vec_c: &'a MatBuild<L, F>,
}

impl<'a, L, F> Operator<F> for ProbSDPOpC<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn size(&self) -> (usize, usize)
    {
        let (n, one) = self.vec_c.size();
        assert_eq!(one, 1);

        (n, 1)
    }

    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        // y = a*vec_f*x + b*y;
        self.vec_c.op(alpha, x, beta, y);
    }

    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        // y = a*vec_f^T*x + b*y;
        self.vec_c.trans_op(alpha, x, beta, y);
    }
}

//

pub struct ProbSDPOpA<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    symmat_f: &'a MatBuild<L, F>,
    mat_a: &'a MatBuild<L, F>,
}

impl<'a, L, F> ProbSDPOpA<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn dim(&self) -> (usize, usize, usize)
    {
        let (sk, n) = self.symmat_f.size();
        let (p, n_) = self.mat_a.size();
        assert_eq!(n, n_);

        (n, sk, p)
    }
}

impl<'a, L, F> Operator<F> for ProbSDPOpA<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn size(&self) -> (usize, usize)
    {
        let (n, sk, p) = self.dim();

        (sk + p, n)
    }

    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (_n, sk, p) = self.dim();

        let (y_sk, y_p) = y.split2(sk, p).unwrap();

        // y_sk = a*symmat_f*x + b*y_sk
        self.symmat_f.op(alpha, x, beta, y_sk);

        // y_p = a*mat_a*x + b*y_p
        self.mat_a.op(alpha, x, beta, y_p);
    }

    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (_n, sk, p) = self.dim();

        let (x_sk, x_p) = x.split2(sk, p).unwrap();

        // y = a*symmat_f^T*x_sk + a*mat_a^T*x_p + b*y
        self.symmat_f.trans_op(alpha, x_sk, beta, y);
        self.mat_a.trans_op(alpha, x_p, F::one(), y);
    }
}

//

pub struct ProbSDPOpB<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    symvec_f_n: &'a MatBuild<L, F>,
    vec_b: &'a MatBuild<L, F>,
}

impl<'a, L, F> ProbSDPOpB<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn dim(&self) -> (usize, usize, usize)
    {
        let (sk, n) = self.symvec_f_n.size();
        let (p, one) = self.vec_b.size();
        assert_eq!(one, 1);

        (n, sk, p)
    }
}

impl<'a, L, F> Operator<F> for ProbSDPOpB<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn size(&self) -> (usize, usize)
    {
        let (_n, sk, p) = self.dim();

        (sk + p, 1)
    }

    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (_n, sk, p) = self.dim();

        let (y_sk, y_p) = y.split2(sk, p).unwrap();

        // y_sk = a*-symmat_f*x + b*y_sk
        self.symvec_f_n.op(-alpha, x, beta, y_sk);

        // y_p = a*vec_b*x + b*y_p
        self.vec_b.op(alpha, x, beta, y_p);
    }

    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (_n, sk, p) = self.dim();

        let (x_sk, x_p) = x.split2(sk, p).unwrap();

        // y = a*-symvec_f_n^T*x_sk + a*vec_b^T*x_p + b*y
        self.symvec_f_n.trans_op(-alpha, x_sk, beta, y);
        self.vec_b.trans_op(alpha, x_p, F::one(), y);
    }
}

//

pub struct ProbSDPCone<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    sk: usize,
    p: usize,
    cone_psd: ConePSD<'a, L, F>,
    cone_zero: ConeZero<F>,
}

impl<'a, L, F> Cone<F> for ProbSDPCone<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn proj(&mut self, dual_cone: bool, x: &mut[F]) -> Result<(), ()>
    {
        let (sk, p) = (self.sk, self.p);
        let (x_sk, x_p) = x.split2(sk, p).unwrap();

        self.cone_psd.proj(dual_cone, x_sk)?;
        self.cone_zero.proj(dual_cone, x_p)?;
        Ok(())
    }

    fn product_group<G: Fn(&mut[F]) + Copy>(&self, dp_tau: &mut[F], group: G)
    {
        let (sk, p) = (self.sk, self.p);
        let (t_sk, t_p) = dp_tau.split2(sk, p).unwrap();

        self.cone_psd.product_group(t_sk, group);
        self.cone_zero.product_group(t_p, group);
    }
    
}

//

/// Semidefinite program
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
/// 
/// The problem is
/// \\[
/// \begin{array}{ll}
/// {\rm minimize} & c^Tx \\\\
/// {\rm subject \ to} & \sum_{i=0}^{n - 1} x_i F_i + F_n \preceq 0 \\\\
/// & A x = b,
/// \end{array}
/// \\]
/// where
/// - variables \\( x \in {\bf R}^n \\)
/// - \\( c \in {\bf R}^n \\)
/// - \\( F_j \in {\bf S}^k \\) for \\( j = 0, \ldots, n \\)
/// - \\( A \in {\bf R}^{p \times n},\ b \in {\bf R}^p \\).
/// 
/// This is already a conic problem and can be reformulated as follows:
/// \\[
/// \begin{array}{ll}
/// {\rm minimize} & c^Tx \\\\
/// {\rm subject \ to} &
///   \left[ \begin{array}{ccc}
///   {\rm vec}(F_0) & \cdots & {\rm vec}(F_{n - 1}) \\\\
///   & A &
///   \end{array} \right]
///   x
///   + s =
///   \left[ \begin{array}{c}
///   -{\rm vec}(F_n) \\\\ b
///   \end{array} \right] \\\\
/// & s \in {\rm vec}({\bf S}_+^k) \times \lbrace 0 \rbrace^p.
/// \end{array}
/// \\]
/// 
/// \\( {\rm vec}(X) = (X_{11}\ \sqrt2 X_{12}\ X_{22}\ \sqrt2 X_{13}\ \sqrt2 X_{23}\ X_{33}\ \cdots)^T \\)
/// which extracts and scales the upper-triangular part of a symmetric matrix X in column-wise.
/// [`ConePSD`] is used for \\( {\rm vec}({\bf S}_+^k) \\).
pub struct ProbSDP<L, F>
where L: LinAlgEx<F>, F: Float
{
    vec_c: MatBuild<L, F>,
    mat_a: MatBuild<L, F>,
    vec_b: MatBuild<L, F>,

    symmat_f: MatBuild<L, F>,
    symvec_f_n: MatBuild<L, F>,

    eps_zero: F,
    w_cone_psd: Vec<F>,
    w_solver: Vec<F>,
}

impl<L, F> ProbSDP<L, F>
where L: LinAlgEx<F>, F: Float
{
    /// Creates a SDP with given data.
    /// 
    /// Returns a [`ProbSDP`] instance.
    /// * `vec_c` is \\(c\\).
    /// * `syms_f` is \\(F_0, \\ldots, F_n\\) each of which shall belong to [`crate::operator::MatType::SymPack`].
    /// * `mat_a` is \\(A\\).
    /// * `vec_b` is \\(b\\).
    /// * `eps_zero` should be the same value as [`crate::solver::SolverParam::eps_zero`].
    pub fn new(
        vec_c: MatBuild<L, F>,
        mut syms_f: Vec<MatBuild<L, F>>,
        mat_a: MatBuild<L, F>, vec_b: MatBuild<L, F>,
        eps_zero: F) -> Self
    {
        let n = vec_c.size().0;
        let p = vec_b.size().0;

        assert_eq!(vec_c.size(), (n, 1));
        assert_eq!(syms_f.len(), n + 1);
        let k = syms_f[0].size().0;
        for sym_f in syms_f.iter() {
            assert!(sym_f.is_sympack());
            assert_eq!(sym_f.size(), (k, k));
        }
        assert_eq!(mat_a.size(), (p, n));
        assert_eq!(vec_b.size(), (p, 1));

        let f1 = F::one();
        let f2 = f1 + f1;
        let fsqrt2 = f2.sqrt();
    
        for sym_f in syms_f.iter_mut() {
            sym_f.set_scale_nondiag(fsqrt2);
            sym_f.set_reshape_colvec();
        }

        let symvec_f_n = syms_f.pop().unwrap();
        let sk = symvec_f_n.size().0;

        let symmat_f = MatBuild::new(MatType::General(sk, n))
                       .by_fn(|r, c| { syms_f[c][(r, 0)] });

        ProbSDP {
            vec_c,
            mat_a,
            vec_b,
            symmat_f,
            symvec_f_n,
            eps_zero,
            w_cone_psd: Vec::new(),
            w_solver: Vec::new(),
        }
    }

    /// Generates the problem data structures to be fed to [`crate::solver::Solver::solve`].
    /// 
    /// Returns a tuple of operators, a cone and a work slice.
    pub fn problem(&mut self) -> (ProbSDPOpC<L, F>, ProbSDPOpA<L, F>, ProbSDPOpB<L, F>, ProbSDPCone<'_, L, F>, &mut[F])
    {
        let p = self.vec_b.size().0;
        let sk = self.symvec_f_n.size().0;

        let f0 = F::zero();

        let op_c = ProbSDPOpC {
            vec_c: &self.vec_c,
        };
        let op_a = ProbSDPOpA {
            symmat_f: &self.symmat_f,
            mat_a: &self.mat_a,
        };
        let op_b = ProbSDPOpB {
            symvec_f_n: &self.symvec_f_n,
            vec_b: &self.vec_b,
        };

        self.w_cone_psd.resize(ConePSD::<L, _>::query_worklen(sk + p), f0);
        let cone = ProbSDPCone {
            sk, p,
            cone_psd: ConePSD::new(self.w_cone_psd.as_mut(), self.eps_zero),
            cone_zero: ConeZero::new(),
        };

        self.w_solver.resize(Solver::<L, _>::query_worklen(op_a.size()), f0);

        (op_c, op_a, op_b, cone, self.w_solver.as_mut())
    }
}
