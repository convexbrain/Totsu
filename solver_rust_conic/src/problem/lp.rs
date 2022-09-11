use std::prelude::v1::*;
use num_traits::{Zero, One};
use crate::solver::Solver;
use crate::linalg::{SliceLike, LinAlgEx};
use crate::operator::{Operator, MatOp, MatBuild};
use crate::cone::{Cone, ConeRPos, ConeZero};
use crate::{splitm, splitm_mut};

//

pub struct ProbLPOpC<'a, L: LinAlgEx>
{
    vec_c: MatOp<'a, L>,
}

impl<'a, L: LinAlgEx> Operator<L> for ProbLPOpC<'a, L>
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

pub struct ProbLPOpA<'a, L: LinAlgEx>
{
    mat_g: MatOp<'a, L>,
    mat_a: MatOp<'a, L>,
}

impl<'a, L: LinAlgEx> ProbLPOpA<'a, L>
{
    fn dim(&self) -> (usize, usize, usize)
    {
        let (m, n) = self.mat_g.size();
        let (p, n_) = self.mat_a.size();
        assert_eq!(n, n_);

        (n, m, p)
    }
}

impl<'a, L: LinAlgEx> Operator<L> for ProbLPOpA<'a, L>
{
    fn size(&self) -> (usize, usize)
    {
        let (n, m, p) = self.dim();

        (m + p, n)
    }

    fn op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        let (_n, m, p) = self.dim();

        splitm_mut!(y, (y_m; m), (y_p; p));

        // y_m = a*mat_g*x + b*y_m
        self.mat_g.op(alpha, x, beta, &mut y_m);

        // y_p = a*mat_a*x + b*y_p
        self.mat_a.op(alpha, x, beta, &mut y_p);
    }

    fn trans_op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        let (_n, m, p) = self.dim();

        splitm!(x, (x_m; m), (x_p; p));

        // y = a*mat_g^T*x_m + a*mat_a^T*x_p + b*y
        self.mat_g.trans_op(alpha, &x_m, beta, y);
        self.mat_a.trans_op(alpha, &x_p, L::F::one(), y);
    }

    fn absadd_cols(&self, tau: &mut L::Sl)
    {
        self.mat_g.absadd_cols(tau);
        self.mat_a.absadd_cols(tau);
    }

    fn absadd_rows(&self, sigma: &mut L::Sl)
    {
        let (_n, m, p) = self.dim();

        splitm_mut!(sigma, (sigma_m; m), (sigma_p; p));

        self.mat_g.absadd_rows(&mut sigma_m);
        self.mat_a.absadd_rows(&mut sigma_p);
    }
}

//

pub struct ProbLPOpB<'a, L: LinAlgEx>
{
    vec_h: MatOp<'a, L>,
    vec_b: MatOp<'a, L>,
}

impl<'a, L: LinAlgEx> ProbLPOpB<'a, L>
{
    fn dim(&self) -> (usize, usize)
    {
        let (m, one) = self.vec_h.size();
        assert_eq!(one, 1);
        let (p, one) = self.vec_b.size();
        assert_eq!(one, 1);

        (m, p)
    }
}

impl<'a, L: LinAlgEx> Operator<L> for ProbLPOpB<'a, L>
{
    fn size(&self) -> (usize, usize)
    {
        let (m, p) = self.dim();

        (m + p, 1)
    }

    fn op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        let (m, p) = self.dim();

        splitm_mut!(y, (y_m; m), (y_p; p));

        // y_m = a*vec_h*x + b*y_m
        self.vec_h.op(alpha, x, beta, &mut y_m);

        // y_p = a*vec_b*x + b*y_p
        self.vec_b.op(alpha, x, beta, &mut y_p);
    }

    fn trans_op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        let (m, p) = self.dim();

        splitm!(x, (x_m; m), (x_p; p));

        // y = a*vec_h^T*x_m + a*vec_b^T*x_p + b*y
        self.vec_h.trans_op(alpha, &x_m, beta, y);
        self.vec_b.trans_op(alpha, &x_p, L::F::one(), y);
    }

    fn absadd_cols(&self, tau: &mut L::Sl)
    {
        self.vec_h.absadd_cols(tau);
        self.vec_b.absadd_cols(tau);
    }

    fn absadd_rows(&self, sigma: &mut L::Sl)
    {
        let (m, p) = self.dim();

        splitm_mut!(sigma, (sigma_m; m), (sigma_p; p));

        self.vec_h.absadd_rows(&mut sigma_m);
        self.vec_b.absadd_rows(&mut sigma_p);
    }
}

//

pub struct ProbLPCone<L: LinAlgEx>
{
    m: usize,
    p: usize,
    cone_rpos: ConeRPos<L>,
    cone_zero: ConeZero<L>,
}

impl<L: LinAlgEx> Cone<L> for ProbLPCone<L>
{
    fn proj(&mut self, dual_cone: bool, x: &mut L::Sl) -> Result<(), ()>
    {
        let (m, p) = (self.m, self.p);
        splitm_mut!(x, (x_m; m), (x_p; p));

        self.cone_rpos.proj(dual_cone, &mut x_m)?;
        self.cone_zero.proj(dual_cone, &mut x_p)?;
        Ok(())
    }

    fn product_group<G: Fn(&mut L::Sl) + Copy>(&self, dp_tau: &mut L::Sl, group: G)
    {
        let (m, p) = (self.m, self.p);
        splitm_mut!(dp_tau, (t_m; m), (t_p; p));

        self.cone_rpos.product_group(&mut t_m, group);
        self.cone_zero.product_group(&mut t_p, group);
    }
}

//

/// Linear program
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
/// 
/// The problem is
/// \\[
/// \begin{array}{ll}
/// {\rm minimize} & c^Tx + d \\\\
/// {\rm subject \ to} & G x \preceq h \\\\
/// & A x = b,
/// \end{array}
/// \\]
/// where
/// - variables \\( x \in \mathbb{R}^n \\)
/// - \\( c \in \mathbb{R}^n,\ d \in \mathbb{R} \\)
/// - \\( G \in \mathbb{R}^{m \times n},\ h \in \mathbb{R}^m \\)
/// - \\( A \in \mathbb{R}^{p \times n},\ b \in \mathbb{R}^p \\).
/// 
/// In the following, \\( d \\) does not appear since it does not matter.
/// 
/// The representation as a conic linear program is as follows:
/// \\[
/// \begin{array}{ll}
/// {\rm minimize} & c^Tx \\\\
/// {\rm subject \ to} &
///   \left[ \begin{array}{c}
///   G \\\\
///   A
///   \end{array} \right]
///   x + s =
///   \left[ \begin{array}{c}
///   h \\\\
///   b
///   \end{array} \right] \\\\
/// & s \in \mathbb{R}_+^m \times \lbrace 0 \rbrace^n.
/// \end{array}
/// \\]
pub struct ProbLP<L: LinAlgEx>
{
    vec_c: MatBuild<L>,
    mat_g: MatBuild<L>,
    vec_h: MatBuild<L>,
    mat_a: MatBuild<L>,
    vec_b: MatBuild<L>,

    w_solver: Vec<L::F>,
}

impl<L: LinAlgEx> ProbLP<L>
{
    /// Creates a LP with given data.
    /// 
    /// Returns a [`ProbLP`] instance.
    /// * `vec_c` is \\(c\\).
    /// * `mat_g` is \\(G\\).
    /// * `vec_h` is \\(h\\).
    /// * `mat_a` is \\(A\\).
    /// * `vec_b` is \\(b\\).
    pub fn new(
        vec_c: MatBuild<L>,
        mat_g: MatBuild<L>, vec_h: MatBuild<L>,
        mat_a: MatBuild<L>, vec_b: MatBuild<L>) -> Self
    {
        let n = vec_c.size().0;
        let m = vec_h.size().0;
        let p = vec_b.size().0;

        assert_eq!(vec_c.size(), (n, 1));
        assert_eq!(mat_g.size(), (m, n));
        assert_eq!(vec_h.size(), (m, 1));
        assert_eq!(mat_a.size(), (p, n));
        assert_eq!(vec_b.size(), (p, 1));

        ProbLP {
            vec_c,
            mat_g,
            vec_h,
            mat_a,
            vec_b,
            w_solver: Vec::new(),
        }
    }

    /// Generates the problem data structures to be fed to [`crate::solver::Solver::solve`].
    /// 
    /// Returns a tuple of operators, a cone and a work slice.
    pub fn problem(&mut self) -> (ProbLPOpC<L>, ProbLPOpA<L>, ProbLPOpB<L>, ProbLPCone<L>, &mut[L::F])
    {
        let m = self.vec_h.size().0;
        let p = self.vec_b.size().0;

        let f0 = L::F::zero();

        let op_c = ProbLPOpC {
            vec_c: MatOp::from(&self.vec_c),
        };
        let op_a = ProbLPOpA {
            mat_g: MatOp::from(&self.mat_g),
            mat_a: MatOp::from(&self.mat_a),
        };
        let op_b = ProbLPOpB {
            vec_h: MatOp::from(&self.vec_h),
            vec_b: MatOp::from(&self.vec_b),
        };

        let cone = ProbLPCone {
            m, p,
            cone_rpos: ConeRPos::new(),
            cone_zero: ConeZero::new(),
        };

        self.w_solver.resize(Solver::<L>::query_worklen(op_a.size()), f0);

        (op_c, op_a, op_b, cone, self.w_solver.as_mut())
    }
}
