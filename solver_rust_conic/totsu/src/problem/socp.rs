use std::prelude::v1::*;
use num_traits::{Zero, One};
use totsu_core::solver::{Solver, SliceLike, Operator, Cone};
use totsu_core::{LinAlgEx, MatOp, ConeSOC, ConeZero, splitm, splitm_mut};
use crate::MatBuild;

//

pub struct ProbSOCPOpC<'a, L: LinAlgEx>
{
    vec_f: MatOp<'a, L>,
}

impl<'a, L: LinAlgEx> Operator<L> for ProbSOCPOpC<'a, L>
{
    fn size(&self) -> (usize, usize)
    {
        let (n, one) = self.vec_f.size();
        assert_eq!(one, 1);

        (n, 1)
    }

    fn op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        // y = a*vec_f*x + b*y;
        self.vec_f.op(alpha, x, beta, y);
    }

    fn trans_op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        // y = a*vec_f^T*x + b*y;
        self.vec_f.trans_op(alpha, x, beta, y);
    }

    fn absadd_cols(&self, tau: &mut L::Sl)
    {
        self.vec_f.absadd_cols(tau);
    }

    fn absadd_rows(&self, sigma: &mut L::Sl)
    {
        self.vec_f.absadd_rows(sigma);
    }
}

//

pub struct ProbSOCPOpA<'a, L: LinAlgEx>
{
    mats_g: Vec<MatOp<'a, L>>,
    vecs_c: Vec<MatOp<'a, L>>,
    mat_a: MatOp<'a, L>,
}

impl<'a, L: LinAlgEx> Operator<L> for ProbSOCPOpA<'a, L>
{
    fn size(&self) -> (usize, usize)
    {
        let (p, n) = self.mat_a.size();

        let mut ni1_sum = 0;

        for (mat_g, vec_c) in self.mats_g.iter().zip(&self.vecs_c) {
            let (n_, one) = vec_c.size();
            assert_eq!(one, 1);
            assert_eq!(n, n_);
            let (ni, n_) = mat_g.size();
            assert_eq!(n, n_);

            ni1_sum += 1 + ni;
        }

        (ni1_sum + p, n)
    }

    fn op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        let (p, _) = self.mat_a.size();

        let mut done = 0;

        for (mat_g, vec_c) in self.mats_g.iter().zip(&self.vecs_c) {
            let (ni, _) = mat_g.size();

            splitm_mut!(y, (_y_done; done), (y_1; 1), (y_ni; ni));
            done += 1 + ni;

            // y_1 = a*-vec_c^T*x + b*y_1
            vec_c.trans_op(-alpha, x, beta, &mut y_1);

            // y_ni = a*-mat_g*x + b*y_ni
            mat_g.op(-alpha, x, beta, &mut y_ni);
        }

        splitm_mut!(y, (_y_done; done), (y_p; p));

        // y_p = a*mat_a*x + b*y_p
        self.mat_a.op(alpha, x, beta, &mut y_p);
    }

    fn trans_op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        let (p, _) = self.mat_a.size();

        let f1 = L::F::one();

        // y = b*y + ...
        L::scale(beta, y);

        let mut done = 0;

        for (mat_g, vec_c) in self.mats_g.iter().zip(&self.vecs_c) {
            let (ni, _) = mat_g.size();

            splitm!(x, (_x_done; done), (x_1; 1), (x_ni; ni));
            done += 1 + ni;

            // y = ... + a*-vec_c*x_1 + ...
            vec_c.op(-alpha, &x_1, f1, y);

            // y = ... + a*-mat_gc^T*x_ni + ...
            mat_g.trans_op(-alpha, &x_ni, f1, y);
        }

        splitm!(x, (_x_done; done), (x_p; p));

        // y = ... + a*mat_a^T*x_p
        self.mat_a.trans_op(alpha, &x_p, f1, y);
    }

    fn absadd_cols(&self, tau: &mut L::Sl)
    {
        for vec_c in &self.vecs_c {
            vec_c.absadd_rows(tau);
        }
        for mat_g in &self.mats_g {
            mat_g.absadd_cols(tau);
        }
        self.mat_a.absadd_cols(tau);
    }

    fn absadd_rows(&self, sigma: &mut L::Sl)
    {
        let (p, _) = self.mat_a.size();
        
        let mut done = 0;

        for (mat_g, vec_c) in self.mats_g.iter().zip(&self.vecs_c) {
            let (ni, _) = mat_g.size();

            splitm_mut!(sigma, (_sigma_done; done), (sigma_1; 1), (sigma_ni; ni));
            done += 1 + ni;

            vec_c.absadd_cols(&mut sigma_1);
            mat_g.absadd_rows(&mut sigma_ni);
        }

        splitm_mut!(sigma, (_sigma_done; done), (sigma_p; p));

        self.mat_a.absadd_rows(&mut sigma_p);
    }
}

//

pub struct ProbSOCPOpB<'a, L: LinAlgEx>
{
    vecs_h: Vec<MatOp<'a, L>>,
    scls_d: &'a[L::F],
    abssum_scls_d: L::F,
    vec_b: MatOp<'a, L>,
}

impl<'a, L: LinAlgEx> Operator<L> for ProbSOCPOpB<'a, L>
{
    fn size(&self) -> (usize, usize)
    {
        let (p, one) = self.vec_b.size();
        assert_eq!(one, 1);

        let mut ni1_sum = 0;

        for vec_h in &self.vecs_h {
            let (ni, one) = vec_h.size();
            assert_eq!(one, 1);

            ni1_sum += 1 + ni;
        }

        (ni1_sum + p, 1)
    }

    fn op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        let (p, _) = self.vec_b.size();
        
        let mut done = 0;

        for (vec_h, scl_d) in self.vecs_h.iter().zip(self.scls_d) {
            let (ni, _) = vec_h.size();

            splitm_mut!(y, (_y_done; done), (y_1; 1), (y_ni; ni));
            done += 1 + ni;

            // y_1 = a*scl_d*x + b*y_1
            L::scale(beta, &mut y_1);
            L::add(alpha * *scl_d, x, &mut y_1);

            // y_ni = a*vec_h*x + b*y_ni
            vec_h.op(alpha, x, beta, &mut y_ni);
        }

        splitm_mut!(y, (_y_done; done), (y_p; p));

        // y_p = a*vec_b*x + b*y_p
        self.vec_b.op(alpha, x, beta, &mut y_p);
    }

    fn trans_op(&self, alpha: L::F, x: &L::Sl, beta: L::F, y: &mut L::Sl)
    {
        let (p, _) = self.vec_b.size();
        
        let f1 = L::F::one();

        // y = b*y + ...
        L::scale(beta, y);

        let mut done = 0;

        for (vec_h, scl_d) in self.vecs_h.iter().zip(self.scls_d) {
            let (ni, _) = vec_h.size();

            splitm!(x, (_x_done; done), (x_1; 1), (x_ni; ni));
            done += 1 + ni;

            // y = ... + a*scl_d*x_1 + ...
            L::add(alpha * *scl_d, &x_1, y);

            // y = ... + a*vec_h^T*x_ni + ...
            vec_h.trans_op(alpha, &x_ni, f1, y);
        }

        splitm!(x, (_x_done; done), (x_p; p));

        // y = ... + a*vec_b^T*x_p
        self.vec_b.trans_op(alpha, &x_p, f1, y);
    }

    fn absadd_cols(&self, tau: &mut L::Sl)
    {
        let val_tau = tau.get(0) + self.abssum_scls_d;
        tau.set(0, val_tau);

        for vec_h in &self.vecs_h {
            vec_h.absadd_cols(tau);
        }
        self.vec_b.absadd_cols(tau);
    }

    fn absadd_rows(&self, sigma: &mut L::Sl)
    {
        let (p, _) = self.vec_b.size();

        let mut done = 0;

        for (vec_h, scl_d) in self.vecs_h.iter().zip(self.scls_d) {
            let (ni, _) = vec_h.size();

            splitm_mut!(sigma, (_sigma_done; done), (sigma_1; 1), (sigma_ni; ni));
            done += 1 + ni;

            let val_sigma_1 = sigma_1.get(0) + *scl_d;
            sigma_1.set(0, val_sigma_1);
            vec_h.absadd_rows(&mut sigma_ni);
        }

        splitm_mut!(sigma, (_sigma_done; done), (sigma_p; p));

        self.vec_b.absadd_rows(&mut sigma_p);
    }
}

//

pub struct ProbSOCPCone<'a, L: LinAlgEx>
{
    p: usize,
    mats_g: Vec<MatOp<'a, L>>,
    cone_soc: ConeSOC<L>,
    cone_zero: ConeZero<L>,
}

impl<'a, L: LinAlgEx> Cone<L> for ProbSOCPCone<'a, L>
{
    fn proj(&mut self, dual_cone: bool, x: &mut L::Sl) -> Result<(), ()>
    {
        let mut done = 0;

        for mat_g in &self.mats_g {
            let ni = mat_g.size().0;

            splitm_mut!(x, (_x_done; done), (x_ni1; 1 + ni));
            done += 1 + ni;

            self.cone_soc.proj(dual_cone, &mut x_ni1)?;
        }

        splitm_mut!(x, (_x_done; done), (x_p; self.p));

        self.cone_zero.proj(dual_cone, &mut x_p)?;
        Ok(())
    }

    fn product_group<G: Fn(&mut L::Sl) + Copy>(&self, dp_tau: &mut L::Sl, group: G)
    {
        let mut done = 0;

        for mat_g in &self.mats_g {
            let ni = mat_g.size().0;

            splitm_mut!(dp_tau, (_t_done; done), (t_ni1; 1 + ni));
            done += 1 + ni;

            self.cone_soc.product_group(&mut t_ni1, group);
        }

        splitm_mut!(dp_tau, (_t_done; done), (t_p; self.p));

        self.cone_zero.product_group(&mut t_p, group);
    }
}

//

/// Second-order cone program
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
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
pub struct ProbSOCP<L: LinAlgEx>
{
    vec_f: MatBuild<L>,
    mats_g: Vec<MatBuild<L>>,
    vecs_h: Vec<MatBuild<L>>,
    vecs_c: Vec<MatBuild<L>>,
    scls_d: Vec<L::F>,
    mat_a: MatBuild<L>,
    vec_b: MatBuild<L>,

    w_solver: Vec<L::F>,
}

impl<L: LinAlgEx> ProbSOCP<L>
{
    /// Creates a SOCP with given data.
    /// 
    /// Returns the [`ProbSOCP`] instance.
    /// * `vec_f` is \\(f\\).
    /// * `mats_g` is \\(G_0, \\ldots, G_{m-1}\\).
    /// * `vecs_h` is \\(h_0, \\ldots, h_{m-1}\\).
    /// * `vecs_c` is \\(c_0, \\ldots, c_{m-1}\\).
    /// * `scls_d` is \\(d_0, \\ldots, d_{m-1}\\).
    /// * `mat_a` is \\(A\\).
    /// * `vec_b` is \\(b\\).
    pub fn new(
        vec_f: MatBuild<L>,
        mats_g: Vec<MatBuild<L>>, vecs_h: Vec<MatBuild<L>>,
        vecs_c: Vec<MatBuild<L>>, scls_d: Vec<L::F>,
        mat_a: MatBuild<L>, vec_b: MatBuild<L>) -> Self
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
    
    /// Generates the problem data structures to be fed to [`Solver::solve`].
    /// 
    /// Returns a tuple of operators, a cone and a work slice.
    pub fn problem(&mut self) -> (ProbSOCPOpC<L>, ProbSOCPOpA<L>, ProbSOCPOpB<L>, ProbSOCPCone<L>, &mut[L::F])
    {
        let p = self.vec_b.size().0;

        let op_c = ProbSOCPOpC {
            vec_f: self.vec_f.as_op(),
        };
        let op_a = ProbSOCPOpA {
            mats_g: self.mats_g.iter().map(|m| m.as_op()).collect(),
            vecs_c: self.vecs_c.iter().map(|m| m.as_op()).collect(),
            mat_a: self.mat_a.as_op(),
        };
        let op_b = ProbSOCPOpB {
            vecs_h: self.vecs_h.iter().map(|m| m.as_op()).collect(),
            scls_d: &self.scls_d,
            abssum_scls_d: L::abssum(&L::Sl::new_ref(&self.scls_d), 1),
            vec_b: self.vec_b.as_op(),
        };

        let cone = ProbSOCPCone {
            p,
            mats_g: self.mats_g.iter().map(|m| m.as_op()).collect(),
            cone_soc: ConeSOC::new(),
            cone_zero: ConeZero::new(),
        };

        self.w_solver.resize(Solver::<L>::query_worklen(op_a.size()), L::F::zero());

        (op_c, op_a, op_b, cone, self.w_solver.as_mut())
    }
}
