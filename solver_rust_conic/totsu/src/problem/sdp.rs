use num_traits::{Float, Zero, One};
use totsu_core::solver::{Solver, SliceLike, Operator, Cone};
use totsu_core::{LinAlgEx, MatType, MatOp, ConePSD, ConeZero, splitm, splitm_mut};
use crate::MatBuild;

//

pub struct ProbSDPOpC<'a, L: LinAlgEx>
{
    vec_c: MatOp<'a, L>,
}

impl<'a, L: LinAlgEx> Operator<L> for ProbSDPOpC<'a, L>
{
    fn size(&self) -> (usize, usize)
    {
        let (n, one) = self.vec_c.size();
        assert_eq!(one, 1);

        (n, 1)
    }

    fn op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        // y = a*vec_c*x + b*y;
        self.vec_c.op(alpha, x, beta, y);
    }

    fn trans_op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        // y = a*vec_c^T*x + b*y;
        self.vec_c.trans_op(alpha, x, beta, y);
    }

    fn absadd_cols(&self, tau: &mut L::Sl)
    {
        self.vec_c.absadd_cols(tau);
    }

    fn absadd_rows(&self, sigma: &mut L::Sl)
    {
        self.vec_c.absadd_rows(sigma);
    }
}

//

pub struct ProbSDPOpA<'a, L: LinAlgEx>
{
    symmat_f: MatOp<'a, L>,
    mat_a: MatOp<'a, L>,
}

impl<'a, L: LinAlgEx> ProbSDPOpA<'a, L>
{
    fn dim(&self) -> (usize, usize, usize)
    {
        let (sk, n) = self.symmat_f.size();
        let (p, n_) = self.mat_a.size();
        assert_eq!(n, n_);

        (n, sk, p)
    }
}

impl<'a, L: LinAlgEx> Operator<L> for ProbSDPOpA<'a, L>
{
    fn size(&self) -> (usize, usize)
    {
        let (n, sk, p) = self.dim();

        (sk + p, n)
    }

    fn op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        let (_n, sk, p) = self.dim();

        splitm_mut!(y, (y_sk; sk), (y_p; p));

        // y_sk = a*symmat_f*x + b*y_sk
        self.symmat_f.op(alpha, x, beta, &mut y_sk);

        // y_p = a*mat_a*x + b*y_p
        self.mat_a.op(alpha, x, beta, &mut y_p);
    }

    fn trans_op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        let (_n, sk, p) = self.dim();

        splitm!(x, (x_sk; sk), (x_p; p));

        // y = a*symmat_f^T*x_sk + a*mat_a^T*x_p + b*y
        self.symmat_f.trans_op(alpha, &x_sk, beta, y);
        self.mat_a.trans_op(alpha, &x_p, L::F::one(), y);
    }

    fn absadd_cols(&self, tau: &mut L::Sl)
    {
        self.symmat_f.absadd_cols(tau);
        self.mat_a.absadd_cols(tau);
    }

    fn absadd_rows(&self, sigma: &mut L::Sl)
    {
        let (_n, sk, p) = self.dim();

        splitm_mut!(sigma, (sigma_sk; sk), (sigma_p; p));

        self.symmat_f.absadd_rows(&mut sigma_sk);
        self.mat_a.absadd_rows(&mut sigma_p);
    }
}

//

pub struct ProbSDPOpB<'a, L: LinAlgEx>
{
    symvec_f_n: MatOp<'a, L>,
    vec_b: MatOp<'a, L>,
}

impl<'a, L: LinAlgEx> ProbSDPOpB<'a, L>
{
    fn dim(&self) -> (usize, usize, usize)
    {
        let (sk, n) = self.symvec_f_n.size();
        let (p, one) = self.vec_b.size();
        assert_eq!(one, 1);

        (n, sk, p)
    }
}

impl<'a, L: LinAlgEx> Operator<L> for ProbSDPOpB<'a, L>
{
    fn size(&self) -> (usize, usize)
    {
        let (_n, sk, p) = self.dim();

        (sk + p, 1)
    }

    fn op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        let (_n, sk, p) = self.dim();

        splitm_mut!(y, (y_sk; sk), (y_p; p));

        // y_sk = a*-symmat_f*x + b*y_sk
        self.symvec_f_n.op(-alpha, x, beta, &mut y_sk);

        // y_p = a*vec_b*x + b*y_p
        self.vec_b.op(alpha, x, beta, &mut y_p);
    }

    fn trans_op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        let (_n, sk, p) = self.dim();

        splitm!(x, (x_sk; sk), (x_p; p));

        // y = a*-symvec_f_n^T*x_sk + a*vec_b^T*x_p + b*y
        self.symvec_f_n.trans_op(-alpha, &x_sk, beta, y);
        self.vec_b.trans_op(alpha, &x_p, L::F::one(), y);
    }

    fn absadd_cols(&self, tau: &mut L::Sl)
    {
        self.symvec_f_n.absadd_cols(tau);
        self.vec_b.absadd_cols(tau);
    }

    fn absadd_rows(&self, sigma: &mut L::Sl)
    {
        let (_n, sk, p) = self.dim();

        splitm_mut!(sigma, (sigma_sk; sk), (sigma_p; p));

        self.symvec_f_n.absadd_rows(&mut sigma_sk);
        self.vec_b.absadd_rows(&mut sigma_p);
    }
}

//

pub struct ProbSDPCone<'a, L: LinAlgEx>
{
    sk: usize,
    p: usize,
    cone_psd: ConePSD<'a, L>,
    cone_zero: ConeZero<L>,
}

impl<'a, L: LinAlgEx> Cone<L> for ProbSDPCone<'a, L>
{
    fn proj(&mut self, dual_cone: bool, x: &mut L::Sl) -> Result<(), ()>
    {
        let (sk, p) = (self.sk, self.p);

        splitm_mut!(x, (x_sk; sk), (x_p; p));

        self.cone_psd.proj(dual_cone, &mut x_sk)?;
        self.cone_zero.proj(dual_cone, &mut x_p)?;
        Ok(())
    }

    fn product_group<G: Fn(&mut L::Sl) + Copy>(&self, dp_tau: &mut L::Sl, group: G)
    {
        let (sk, p) = (self.sk, self.p);

        splitm_mut!(dp_tau, (t_sk; sk), (t_p; p));

        self.cone_psd.product_group(&mut t_sk, group);
        self.cone_zero.product_group(&mut t_p, group);
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
/// - variables \\( x \in \mathbb{R}^n \\)
/// - \\( c \in \mathbb{R}^n \\)
/// - \\( F_j \in \mathcal{S}^k \\) for \\( j = 0, \ldots, n \\)
/// - \\( A \in \mathbb{R}^{p \times n},\ b \in \mathbb{R}^p \\).
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
/// & s \in {\rm vec}(\mathcal{S}_+^k) \times \lbrace 0 \rbrace^p.
/// \end{array}
/// \\]
/// 
/// \\( {\rm vec}(X) = (X_{11}\ \sqrt2 X_{12}\ X_{22}\ \sqrt2 X_{13}\ \sqrt2 X_{23}\ X_{33}\ \cdots)^T \\)
/// which extracts and scales the upper-triangular part of a symmetric matrix X in column-wise.
/// [`ConePSD`] is used for \\( {\rm vec}(\mathcal{S}_+^k) \\).
pub struct ProbSDP<L: LinAlgEx>
{
    vec_c: MatBuild<L>,
    mat_a: MatBuild<L>,
    vec_b: MatBuild<L>,

    symmat_f: MatBuild<L>,
    symvec_f_n: MatBuild<L>,

    eps_zero: L::F,
    w_cone_psd: Vec<L::F>,
    w_solver: Vec<L::F>,
}

impl<L: LinAlgEx> ProbSDP<L>
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
        vec_c: MatBuild<L>,
        mut syms_f: Vec<MatBuild<L>>,
        mat_a: MatBuild<L>, vec_b: MatBuild<L>,
        eps_zero: L::F) -> Self
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

        let f1 = L::F::one();
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
    pub fn problem(&mut self) -> (ProbSDPOpC<L>, ProbSDPOpA<L>, ProbSDPOpB<L>, ProbSDPCone<'_, L>, &mut[L::F])
    {
        let p = self.vec_b.size().0;
        let sk = self.symvec_f_n.size().0;

        let f0 = L::F::zero();

        let op_c = ProbSDPOpC {
            vec_c: self.vec_c.as_op(),
        };
        let op_a = ProbSDPOpA {
            symmat_f: self.symmat_f.as_op(),
            mat_a: self.mat_a.as_op(),
        };
        let op_b = ProbSDPOpB {
            symvec_f_n: self.symvec_f_n.as_op(),
            vec_b: self.vec_b.as_op(),
        };

        self.w_cone_psd.resize(ConePSD::<L>::query_worklen(sk), f0);
        let cone = ProbSDPCone {
            sk, p,
            cone_psd: ConePSD::new(self.w_cone_psd.as_mut(), self.eps_zero),
            cone_zero: ConeZero::new(),
        };

        self.w_solver.resize(Solver::<L>::query_worklen(op_a.size()), f0);

        (op_c, op_a, op_b, cone, self.w_solver.as_mut())
    }
}
