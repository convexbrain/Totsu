use std::prelude::v1::*;
use num_traits::Float;
use crate::solver::Solver;
use crate::linalg::LinAlgEx;
use crate::operator::{Operator, MatBuild};
use crate::cone::{Cone, ConeSOC, ConeZero};

//

pub struct ProbSOCPOpC<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    vec_f: &'a MatBuild<L, F>,
}

impl<'a, L, F> Operator<F> for ProbSOCPOpC<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn size(&self) -> (usize, usize)
    {
        let (n, one) = self.vec_f.size();
        assert_eq!(one, 1);

        (n, 1)
    }

    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        // y = a*vec_f*x + b*y;
        self.vec_f.op(alpha, x, beta, y);
    }

    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        // y = a*vec_f^T*x + b*y;
        self.vec_f.trans_op(alpha, x, beta, y);
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

pub struct ProbSOCPOpA<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    mats_g: &'a[MatBuild<L, F>],
    vecs_c: &'a[MatBuild<L, F>],
    mat_a: &'a MatBuild<L, F>,
}

impl<'a, L, F> Operator<F> for ProbSOCPOpA<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn size(&self) -> (usize, usize)
    {
        let (p, n) = self.mat_a.size();

        let mut ni1_sum = 0;

        for (mat_g, vec_c) in self.mats_g.iter().zip(self.vecs_c) {
            let (n_, one) = vec_c.size();
            assert_eq!(one, 1);
            assert_eq!(n, n_);
            let (ni, n_) = mat_g.size();
            assert_eq!(n, n_);

            ni1_sum += 1 + ni;
        }

        (ni1_sum + p, n)
    }

    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let mut spl_y = y;

        for (mat_g, vec_c) in self.mats_g.iter().zip(self.vecs_c) {
            let (ni, _) = mat_g.size();

            let (y_1, spl) = spl_y.split_at_mut(1);
            let (y_ni, spl) = spl.split_at_mut(ni);
            spl_y = spl;

            // y_1 = a*-vec_c^T*x + b*y_1
            vec_c.trans_op(-alpha, x, beta, y_1);

            // y_ni = a*-mat_g*x + b*y_ni
            mat_g.op(-alpha, x, beta, y_ni);
        }

        let y_p = spl_y;

        // y_p = a*mat_a*x + b*y_p
        self.mat_a.op(alpha, x, beta, y_p);
    }

    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let mut spl_x = x;

        let f1 = F::one();

        // y = b*y + ...
        L::scale(beta, y);

        for (mat_g, vec_c) in self.mats_g.iter().zip(self.vecs_c) {
            let (ni, _) = mat_g.size();

            let (x_1, spl) = spl_x.split_at(1);
            let (x_ni, spl) = spl.split_at(ni);
            spl_x = spl;

            // y = ... + a*-vec_c*x_1 + ...
            vec_c.op(-alpha, x_1, f1, y);

            // y = ... + a*-mat_gc^T*x_ni + ...
            mat_g.trans_op(-alpha, x_ni, f1, y);
        }

        let x_p = spl_x;

        // y = ... + a*mat_a^T*x_p
        self.mat_a.trans_op(alpha, x_p, f1, y);
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

pub struct ProbSOCPOpB<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    vecs_h: &'a[MatBuild<L, F>],
    scls_d: &'a[F],
    vec_b: &'a MatBuild<L, F>,
}

impl<'a, L, F> Operator<F> for ProbSOCPOpB<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn size(&self) -> (usize, usize)
    {
        let (p, one) = self.vec_b.size();
        assert_eq!(one, 1);

        let mut ni1_sum = 0;

        for vec_h in self.vecs_h {
            let (ni, one) = vec_h.size();
            assert_eq!(one, 1);

            ni1_sum += 1 + ni;
        }

        (ni1_sum + p, 1)
    }

    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let mut spl_y = y;

        for (vec_h, scl_d) in self.vecs_h.iter().zip(self.scls_d) {
            let (ni, _) = vec_h.size();

            let (y_1, spl) = spl_y.split_at_mut(1);
            let (y_ni, spl) = spl.split_at_mut(ni);
            spl_y = spl;

            // y_1 = a*scl_d*x + b*y_1
            L::scale(beta, y_1);
            L::add(alpha * *scl_d, x, y_1);

            // y_ni = a*vec_h*x + b*y_ni
            vec_h.op(alpha, x, beta, y_ni);
        }

        let y_p = spl_y;

        // y_p = a*vec_b*x + b*y_p
        self.vec_b.op(alpha, x, beta, y_p);
    }

    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let mut spl_x = x;

        let f1 = F::one();

        // y = b*y + ...
        L::scale(beta, y);

        for (vec_h, scl_d) in self.vecs_h.iter().zip(self.scls_d) {
            let (ni, _) = vec_h.size();

            let (x_1, spl) = spl_x.split_at(1);
            let (x_ni, spl) = spl.split_at(ni);
            spl_x = spl;

            // y = ... + a*scl_d*x_1 + ...
            L::add(alpha * *scl_d, x_1, y);

            // y = ... + a*vec_h^T*x_ni + ...
            vec_h.trans_op(alpha, x_ni, f1, y);
        }

        let x_p = spl_x;

        // y = ... + a*vec_b^T*x_p
        self.vec_b.trans_op(alpha, x_p, f1, y);
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

pub struct ProbSOCPCone<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    mats_g: &'a[MatBuild<L, F>],
    cone_soc: ConeSOC<L, F>,
    cone_zero: ConeZero<F>,
}

impl<'a, L, F> Cone<F> for ProbSOCPCone<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn proj(&mut self, dual_cone: bool, x: &mut[F]) -> Result<(), ()>
    {
        let mut spl_x = x;

        for mat_g in self.mats_g {
            let ni = mat_g.size().0;
            let (x_ni1, spl) = spl_x.split_at_mut(1 + ni);
            spl_x = spl;

            self.cone_soc.proj(dual_cone, x_ni1)?;
        }

        let x_p = spl_x;

        self.cone_zero.proj(dual_cone, x_p)?;
        Ok(())
    }

    fn product_group<G: Fn(&mut[F]) + Copy>(&self, dp_tau: &mut[F], group: G)
    {
        let mut spl_t = dp_tau;

        for mat_g in self.mats_g {
            let ni = mat_g.size().0;
            let (t_ni1, spl) = spl_t.split_at_mut(1 + ni);
            spl_t = spl;

            self.cone_soc.product_group(t_ni1, group);
        }

        let t_p = spl_t;

        self.cone_zero.product_group(t_p, group);
    }
}

//

/// Second-order cone program
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
/// 
/// The problem is
/// \\[
/// \begin{array}{ll}
/// {\rm minimize} & f^T x \\\\
/// {\rm subject \ to} & \\| G_i x + h_i \\|_2 \le c_i^T x + d_i \quad (i = 0, \ldots, m - 1) \\\\
/// & A x = b,
/// \end{array}
/// \\]
/// where
/// - variables \\( x \in \mathbb{R}^n \\)
/// - \\( f \in \mathbb{R}^n \\)
/// - \\( G_i \in \mathbb{R}^{n_i \times n},\ h_i \in \mathbb{R}^{n_i},\ c_i \in \mathbb{R}^n,\ d_i \in \mathbb{R} \\)
/// - \\( A \in \mathbb{R}^{p \times n},\ b \in \mathbb{R}^p \\).
/// 
/// The representation as a conic linear program is as follows:
/// \\[
/// \begin{array}{ll}
/// {\rm minimize} & f^T x \\\\
/// {\rm subject \ to} &
///   \left[ \begin{array}{c}
///   -c_0^T \\\\ -G_0 \\\\
///   \vdots \\\\
///   -c_{m - 1}^T \\\\ -G_{m - 1} \\\\
///   A
///   \end{array} \right]
///   x + s =
///   \left[ \begin{array}{c}
///   d_0 \\\\ h_0 \\\\
///   \vdots \\\\
///   d_{m - 1} \\\\ h_{m - 1} \\\\
///   b
///   \end{array} \right] \\\\
/// & s \in \mathcal{Q}^{1 + n_0} \times \cdots \times \mathcal{Q}^{1 + n_{m - 1}} \times \lbrace 0 \rbrace^p.
/// \end{array}
/// \\]
/// 
/// \\( \mathcal{Q} \\) is a second-order (or quadratic) cone (see [`ConeSOC`]).
pub struct ProbSOCP<L, F>
where L: LinAlgEx<F>, F: Float
{
    vec_f: MatBuild<L, F>,
    mats_g: Vec<MatBuild<L, F>>,
    vecs_h: Vec<MatBuild<L, F>>,
    vecs_c: Vec<MatBuild<L, F>>,
    scls_d: Vec<F>,
    mat_a: MatBuild<L, F>,
    vec_b: MatBuild<L, F>,

    w_solver: Vec<F>,
}

impl<L, F> ProbSOCP<L, F>
where L: LinAlgEx<F>, F: Float
{
    /// Creates a SOCP with given data.
    /// 
    /// Returns a [`ProbSOCP`] instance.
    /// * `vec_f` is \\(f\\).
    /// * `mats_g` is \\(G_0, \\ldots, G_{m-1}\\).
    /// * `vecs_h` is \\(h_0, \\ldots, h_{m-1}\\).
    /// * `vecs_c` is \\(c_0, \\ldots, c_{m-1}\\).
    /// * `scls_d` is \\(d_0, \\ldots, d_{m-1}\\).
    /// * `mat_a` is \\(A\\).
    /// * `vec_b` is \\(b\\).
    pub fn new(
        vec_f: MatBuild<L, F>,
        mats_g: Vec<MatBuild<L, F>>, vecs_h: Vec<MatBuild<L, F>>,
        vecs_c: Vec<MatBuild<L, F>>, scls_d: Vec<F>,
        mat_a: MatBuild<L, F>, vec_b: MatBuild<L, F>) -> Self
    {
        let n = vec_f.size().0;
        let m = mats_g.len();
        let p = vec_b.size().0;
    
        assert_eq!(mats_g.len(), m);
        assert_eq!(vecs_h.len(), m);
        assert_eq!(vecs_c.len(), m);
        assert_eq!(scls_d.len(), m);
        assert_eq!(vec_f.size(), (n, 1));
        for i in 0.. m {
            let ni = mats_g[i].size().0;
            assert_eq!(mats_g[i].size(), (ni, n));
            assert_eq!(vecs_h[i].size(), (ni, 1));
            assert_eq!(vecs_c[i].size(), (n, 1));
        }
        assert_eq!(mat_a.size(), (p, n));
        assert_eq!(vec_b.size(), (p, 1));

        ProbSOCP {
            vec_f,
            mats_g,
            vecs_h,
            vecs_c,
            scls_d,
            mat_a,
            vec_b,
            w_solver: Vec::new(),
        }
    }
    
    /// Generates the problem data structures to be fed to [`crate::solver::Solver::solve`].
    /// 
    /// Returns a tuple of operators, a cone and a work slice.
    pub fn problem(&mut self) -> (ProbSOCPOpC<L, F>, ProbSOCPOpA<L, F>, ProbSOCPOpB<L, F>, ProbSOCPCone<'_, L, F>, &mut[F])
    {
        let op_c = ProbSOCPOpC {
            vec_f: &self.vec_f,
        };
        let op_a = ProbSOCPOpA {
            mats_g: &self.mats_g,
            vecs_c: &self.vecs_c,
            mat_a: &self.mat_a,
        };
        let op_b = ProbSOCPOpB {
            vecs_h: &self.vecs_h,
            scls_d: &self.scls_d,
            vec_b: &self.vec_b,
        };

        let cone = ProbSOCPCone {
            mats_g: &self.mats_g,
            cone_soc: ConeSOC::new(),
            cone_zero: ConeZero::new(),
        };

        self.w_solver.resize(Solver::<L, _>::query_worklen(op_a.size()), F::zero());

        (op_c, op_a, op_b, cone, self.w_solver.as_mut())
    }
}
