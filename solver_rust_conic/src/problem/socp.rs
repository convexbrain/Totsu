use num::Float;
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
            let (ni, n_) = mat_g.size();
            assert_eq!(n, n_);
            let (n_, one) = vec_c.size();
            assert_eq!(one, 1);
            assert_eq!(n, n_);

            ni1_sum += ni + 1;
        }

        (ni1_sum + p, n)
    }

    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let mut spl_y = y;

        for (mat_g, vec_c) in self.mats_g.iter().zip(self.vecs_c) {
            let (ni, _) = mat_g.size();

            let (y_ni, spl) = spl_y.split_at_mut(ni);
            let (y_1, spl) = spl.split_at_mut(1);
            spl_y = spl;

            // y_ni = a*-mat_g*x + b*y_ni
            mat_g.op(-alpha, x, beta, y_ni);

            // y_1 = a*-vec_c^T*x + b*y_1
            vec_c.trans_op(-alpha, x, beta, y_1);
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

            let (x_ni, spl) = spl_x.split_at(ni);
            let (x_1, spl) = spl.split_at(1);
            spl_x = spl;

            // y = ... + a*-mat_gc^T*x_ni + ...
            mat_g.trans_op(-alpha, x_ni, f1, y);

            // y = ... + a*-vec_c*x_1 + ...
            vec_c.op(-alpha, x_1, f1, y);
        }

        let x_p = spl_x;

        // y = ... + a*mat_a^T*x_p
        self.mat_a.trans_op(alpha, x_p, f1, y);
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

            ni1_sum += ni + 1;
        }

        (ni1_sum + p, 1)
    }

    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let mut spl_y = y;

        for (vec_h, scl_d) in self.vecs_h.iter().zip(self.scls_d) {
            let (ni, _) = vec_h.size();

            let (y_ni, spl) = spl_y.split_at_mut(ni);
            let (y_1, spl) = spl.split_at_mut(1);
            spl_y = spl;

            // y_ni = a*vec_h*x + b*y_ni
            vec_h.op(alpha, x, beta, y_ni);

            // y_1 = a*scl_d*x + b*y_1
            L::scale(beta, y_1);
            L::add(alpha * *scl_d, x, y_1);
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

            let (x_ni, spl) = spl_x.split_at(ni);
            let (x_1, spl) = spl.split_at(1);
            spl_x = spl;

            // y = ... + a*vec_h^T*x_ni + ...
            vec_h.trans_op(alpha, x_ni, f1, y);

            // y = ... + a*scl_d*x_1 + ...
            L::add(alpha * *scl_d, x_1, y);
        }

        let x_p = spl_x;

        // y = ... + a*vec_b^T*x_p
        self.vec_b.trans_op(alpha, x_p, f1, y);
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
    fn proj(&mut self, dual_cone: bool, eps_zero: F, x: &mut[F]) -> Result<(), ()>
    {
        let mut spl_x = x;

        for mat_g in self.mats_g {
            let ni = mat_g.size().0;
            let (x_ni1, spl) = spl_x.split_at_mut(ni + 1);
            spl_x = spl;

            self.cone_soc.proj(dual_cone, eps_zero, x_ni1)?;
        }

        let x_p = spl_x;

        self.cone_zero.proj(dual_cone, eps_zero, x_p)?;
        Ok(())
    }

    fn product_group(&self, dp_tau: &mut[F], group: fn(&mut[F]))
    {
        let mut spl_t = dp_tau;

        for mat_g in self.mats_g {
            let ni = mat_g.size().0;
            let (t_ni1, spl) = spl_t.split_at_mut(ni + 1);
            spl_t = spl;

            self.cone_soc.product_group(t_ni1, group);
        }

        let t_p = spl_t;

        self.cone_zero.product_group(t_p, group);
    }
}

//

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
