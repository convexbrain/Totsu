use num::Float;
use crate::solver::Solver;
use crate::linalg::LinAlgEx;
use crate::operator::{Operator, MatBuild};
use crate::cone::{Cone, ConeRPos, ConeZero};
use crate::utils::*;

//

pub struct ProbLPOpC<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    vec_c: &'a MatBuild<L, F>,
}

impl<'a, L, F> Operator<F> for ProbLPOpC<'a, L, F>
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

pub struct ProbLPOpA<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    mat_g: &'a MatBuild<L, F>,
    mat_a: &'a MatBuild<L, F>,
}

impl<'a, L, F> ProbLPOpA<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn dim(&self) -> (usize, usize, usize)
    {
        let (m, n) = self.mat_g.size();
        let (p, n_) = self.mat_a.size();
        assert_eq!(n, n_);

        (n, m, p)
    }
}

impl<'a, L, F> Operator<F> for ProbLPOpA<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn size(&self) -> (usize, usize)
    {
        let (n, m, p) = self.dim();

        (m + p, n)
    }

    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (_n, m, p) = self.dim();

        let (y_m, y_p) = y.split2(m, p).unwrap();

        // y_m = a*mat_g*x + b*y_m
        self.mat_g.op(alpha, x, beta, y_m);

        // y_p = a*mat_a*x + b*y_p
        self.mat_a.op(alpha, x, beta, y_p);
    }

    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (_n, m, p) = self.dim();

        let (x_m, x_p) = x.split2(m,p).unwrap();

        // y = a*mat_g^T*x_m + a*mat_a^T*x_p + b*y
        self.mat_g.trans_op(alpha, x_m, beta, y);
        self.mat_a.trans_op(alpha, x_p, F::one(), y);
    }
}

//

pub struct ProbLPOpB<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    vec_h: &'a MatBuild<L, F>,
    vec_b: &'a MatBuild<L, F>,
}

impl<'a, L, F> ProbLPOpB<'a, L, F>
where L: LinAlgEx<F>, F: Float
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

impl<'a, L, F> Operator<F> for ProbLPOpB<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn size(&self) -> (usize, usize)
    {
        let (m, p) = self.dim();

        (m + p, 1)
    }

    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (m, p) = self.dim();

        let (y_m, y_p) = y.split2(m, p).unwrap();

        // y_m = a*vec_h*x + b*y_m
        self.vec_h.op(alpha, x, beta, y_m);

        // y_p = a*vec_b*x + b*y_p
        self.vec_b.op(alpha, x, beta, y_p);
    }

    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (m, p) = self.dim();

        let (x_m, x_p) = x.split2(m, p).unwrap();

        // y = a*vec_h^T*x_m + a*vec_b^T*x_p + b*y
        self.vec_h.trans_op(alpha, x_m, beta, y);
        self.vec_b.trans_op(alpha, x_p, F::one(), y);
    }
}

//

pub struct ProbLPCone<F>
where F: Float
{
    m: usize,
    p: usize,
    cone_rpos: ConeRPos<F>,
    cone_zero: ConeZero<F>,
}

impl<F> Cone<F> for ProbLPCone<F>
where F: Float
{
    fn proj(&mut self, dual_cone: bool, eps_zero: F, x: &mut[F]) -> Result<(), ()>
    {
        let (m, p) = (self.m, self.p);
        let (x_m, x_p) = x.split2(m, p).unwrap();

        self.cone_rpos.proj(dual_cone, eps_zero, x_m)?;
        self.cone_zero.proj(dual_cone, eps_zero, x_p)?;
        Ok(())
    }

    fn product_group<G: Fn(&mut[F]) + Copy>(&self, dp_tau: &mut[F], group: G)
    {
        let (m, p) = (self.m, self.p);
        let (t_m, t_p) = dp_tau.split2(m, p).unwrap();

        self.cone_rpos.product_group(t_m, group);
        self.cone_zero.product_group(t_p, group);
    }
}

//

pub struct ProbLP<L, F>
where L: LinAlgEx<F>, F: Float
{
    vec_c: MatBuild<L, F>,
    mat_g: MatBuild<L, F>,
    vec_h: MatBuild<L, F>,
    mat_a: MatBuild<L, F>,
    vec_b: MatBuild<L, F>,

    w_solver: Vec<F>,
}

impl<L, F> ProbLP<L, F>
where L: LinAlgEx<F>, F: Float
{
    pub fn new(
        vec_c: MatBuild<L, F>,
        mat_g: MatBuild<L, F>, vec_h: MatBuild<L, F>,
        mat_a: MatBuild<L, F>, vec_b: MatBuild<L, F>) -> Self
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

    pub fn problem(&mut self) -> (ProbLPOpC<L, F>, ProbLPOpA<L, F>, ProbLPOpB<L, F>, ProbLPCone<F>, &mut[F])
    {
        let m = self.vec_h.size().0;
        let p = self.vec_b.size().0;

        let f0 = F::zero();

        let op_c = ProbLPOpC {
            vec_c: &self.vec_c,
        };
        let op_a = ProbLPOpA {
            mat_g: &self.mat_g,
            mat_a: &self.mat_a,
        };
        let op_b = ProbLPOpB {
            vec_h: &self.vec_h,
            vec_b: &self.vec_b,
        };

        let cone = ProbLPCone {
            m, p,
            cone_rpos: ConeRPos::new(),
            cone_zero: ConeZero::new(),
        };

        self.w_solver.resize(Solver::<L, _>::query_worklen(op_a.size()), f0);

        (op_c, op_a, op_b, cone, self.w_solver.as_mut())
    }
}
