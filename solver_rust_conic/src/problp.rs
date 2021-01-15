use crate::matbuild::MatBuild;
use crate::solver::{Operator, Cone, SolverError, Solver};
use crate::linalgex::LinAlgEx;
use crate::cone::{ConeRPos, ConeZero};
use num::Float;

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

        let (y_m, y) = y.split_at_mut(m);
        let (y_p, _) = y.split_at_mut(p);

        // y_m = a*mat_g*x + b*y_m
        self.mat_g.op(alpha, x, beta, y_m);

        // y_p = a*mat_a*x + b*y_p
        self.mat_a.op(alpha, x, beta, y_p);
    }

    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (_n, m, p) = self.dim();

        let (x_m, x) = x.split_at(m);
        let (x_p, _) = x.split_at(p);

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

        let (y_m, y) = y.split_at_mut(m);
        let (y_p, _) = y.split_at_mut(p);

        // y_m = a*vec_h*x + b*y_m
        self.vec_h.op(alpha, x, beta, y_m);

        // y_p = a*vec_b*x + b*y_p
        self.vec_b.op(alpha, x, beta, y_p);
    }

    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (m, p) = self.dim();

        let (x_m, x) = x.split_at(m);
        let (x_p, _) = x.split_at(p);

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
    fn proj(&mut self, eps_zero: F, x: &mut[F]) -> Result<(), SolverError>
    {
        let (m, p) = (self.m, self.p);
        let (x_m, x) = x.split_at_mut(m);
        let (x_p, _) = x.split_at_mut(p);

        self.cone_rpos.proj(eps_zero, x_m)?;
        self.cone_zero.proj(eps_zero, x_p)?;
        Ok(())
    }

    fn dual_proj(&mut self, eps_zero: F, x: &mut[F]) -> Result<(), SolverError>
    {
        let (m, p) = (self.m, self.p);
        let (x_m, x) = x.split_at_mut(m);
        let (x_p, _) = x.split_at_mut(p);

        self.cone_rpos.dual_proj(eps_zero, x_m)?;
        self.cone_zero.dual_proj(eps_zero, x_p)?;
        Ok(())
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

//

#[test]
fn test_lp1()
{
    use crate::stdlogger::PrintLogger;
    use crate::matop::MatType;
    use crate::f64lapack::F64LAPACK;
    use crate::floatgeneric::FloatGeneric;
    
    type _LA = F64LAPACK;
    type LA = FloatGeneric<f64>;
    type ASolver = Solver<LA, f64>;
    type AProbLP = ProbLP<LA, f64>;
    type AMatBuild = MatBuild<LA, f64>;

    let n = 1;
    let m = 2;
    let p = 0;

    let vec_c = AMatBuild::new(MatType::General(n, 1)).iter_colmaj(&[
        1.,
    ]);

    // x <= b, x >= c
    let mat_g = AMatBuild::new(MatType::General(m, n)).iter_rowmaj(&[
        1., -1.,
    ]);
    let vec_h = AMatBuild::new(MatType::General(m, 1)).iter_colmaj(&[
        -5., // b
        -(10.)  // -c
    ]);

    let mat_a = AMatBuild::new(MatType::General(p, n));

    let vec_b = AMatBuild::new(MatType::General(p, 1));


    let s = ASolver::new();
    println!("{:?}", s.par);
    let mut lp = AProbLP::new(vec_c, mat_g, vec_h, mat_a, vec_b);
    let rslt = s.solve(lp.problem(), PrintLogger).unwrap_err();
    println!("{:?}", rslt);
    
    assert_eq!(rslt, SolverError::Infeasible);
}
