use num::Float;
use crate::solver::{SolverError, Solver};
use crate::linalg::LinAlgEx;
use crate::operator::{Operator, MatBuild};
use crate::cone::{Cone, ConePSD, ConeRPos, ConeZero};

//

pub struct ProbQPOpC<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    vec_q: &'a MatBuild<L, F>,
}

impl<'a, L, F> ProbQPOpC<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn dim(&self) -> usize
    {
        let (n, one) = self.vec_q.size();
        assert_eq!(one, 1);
        n
    }
}

impl<'a, L, F> Operator<F> for ProbQPOpC<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn size(&self) -> (usize, usize)
    {
        let n = self.dim();

        (n + 1, 1)
    }

    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let n = self.dim();
        let (y_n, y_1) = y.split_at_mut(n);

        // y_n = a*vec_q*x + b*y_n;
        self.vec_q.op(alpha, x, beta, y_n);

        // y_1 = a*1*x + b*y_1;
        L::scale(beta, y_1);
        L::add(alpha, x, y_1);
    }

    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let n = self.dim();
        let (x_n, x_1) = x.split_at(n);

        // y = a*vec_q^T*x_n + a*1*x_1 + b*y;
        self.vec_q.trans_op(alpha, x_n, beta, y);
        L::add(alpha, x_1, y);
    }
}

//

pub struct ProbQPOpA<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    sym_p: &'a MatBuild<L, F>,
    mat_g: &'a MatBuild<L, F>,
    mat_a: &'a MatBuild<L, F>,
}

impl<'a, L, F> ProbQPOpA<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn dim(&self) -> (usize, usize, usize, usize)
    {
        let (n, n_) = self.sym_p.size();
        assert_eq!(n, n_);
        let (m, n_) = self.mat_g.size();
        assert_eq!(n, n_);
        let (p, n_) = self.mat_a.size();
        assert_eq!(n, n_);

        (n, n * (n + 1) / 2, m, p)
    }
}

impl<'a, L, F> Operator<F> for ProbQPOpA<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn size(&self) -> (usize, usize)
    {
        let (n, sn, m, p) = self.dim();

        ((sn + n + 1) + m + p, n + 1)
    }

    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (n, sn, m, p) = self.dim();
        let (x_n, x) = x.split_at(n);
        let (x_1, _) = x.split_at(1);
        let (y_sn, y) = y.split_at_mut(sn);
        let (y_n, y) = y.split_at_mut(n);
        let (y_1, y) = y.split_at_mut(1);
        let (y_m, y) = y.split_at_mut(m);
        let (y_p, _) = y.split_at_mut(p);

        let f2 = F::one() + F::one();
        let fsqrt2 = f2.sqrt();
        
        // y_sn = 0*x_n + 0*x_1 + b*y_sn
        L::scale(beta, y_sn);

        // y_n = a*-sqrt(2)*sym_p*x_n + 0*x_1 + b*y_n
        self.sym_p.op(-alpha * fsqrt2, x_n, beta, y_n);

        // y_1 = 0*x_n + a*-2*x_1 + b*y_1
        L::scale(beta, y_1);
        L::add(-f2 * alpha, x_1, y_1);

        // y_m = a*mat_g*x_n + 0*x_1 + b*y_m
        self.mat_g.op(alpha, x_n, beta, y_m);

        // y_p = a*mat_a*x_n + 0*x_1 + b*y_p
        self.mat_a.op(alpha, x_n, beta, y_p);
    }

    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (n, sn, m, p) = self.dim();
        let (_x_sn, x) = x.split_at(sn);
        let (x_n, x) = x.split_at(n);
        let (x_1, x) = x.split_at(1);
        let (x_m, x) = x.split_at(m);
        let (x_p, _) = x.split_at(p);
        let (y_n, y) = y.split_at_mut(n);
        let (y_1, _) = y.split_at_mut(1);

        let f1 = F::one();
        let f2 = f1 + f1;
        let fsqrt2 = f2.sqrt();
        
        // y_n = 0*x_sn + a*-sqrt(2)*sym_p*x_n + 0*x_1 + a*mat_g^T*x_m + a*mat_a^T*x_p + b*y_n
        self.sym_p.trans_op(-alpha * fsqrt2, x_n, beta, y_n);
        self.mat_g.trans_op(alpha, x_m, f1, y_n);
        self.mat_a.trans_op(alpha, x_p, f1, y_n);

        // y_1 = 0*x_sn + 0*x_n + a*-2*x_1 + 0*x_m + 0*x_p + b*y_1
        L::scale(beta, y_1);
        L::add(-f2 * alpha, x_1, y_1);
    }
}

//

pub struct ProbQPOpB<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    n: usize,
    symvec_p: &'a MatBuild<L, F>,
    vec_h: &'a MatBuild<L, F>,
    vec_b: &'a MatBuild<L, F>,
}

impl<'a, L, F> ProbQPOpB<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn dim(&self) -> (usize, usize, usize, usize)
    {
        let (sn, one) = self.symvec_p.size();
        assert_eq!(self.n * (self.n + 1) / 2, sn);
        assert_eq!(one, 1);
        let (m, one) = self.vec_h.size();
        assert_eq!(one, 1);
        let (p, one) = self.vec_b.size();
        assert_eq!(one, 1);

        (self.n, sn, m, p)
    }
}

impl<'a, L, F> Operator<F> for ProbQPOpB<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn size(&self) -> (usize, usize)
    {
        let (n, sn, m, p) = self.dim();

        ((sn + n + 1) + m + p, 1)
    }

    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (n, sn, m, p) = self.dim();
        let (y_sn, y) = y.split_at_mut(sn);
        let (y_n1, y) = y.split_at_mut(n + 1);
        let (y_m, y) = y.split_at_mut(m);
        let (y_p, _) = y.split_at_mut(p);

        // y_sn = a*symvec_p*x + b*y_sn
        self.symvec_p.op(alpha, x, beta, y_sn);

        // y_n1 = 0*x + b*y_n1
        L::scale(beta, y_n1);

        // y_m = a*vec_h*x + b*y_m
        self.vec_h.op(alpha, x, beta, y_m);

        // y_p = a*vec_b*x + b*y_p
        self.vec_b.op(alpha, x, beta, y_p);
    }

    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (n, sn, m, p) = self.dim();
        let (x_sn, x) = x.split_at(sn);
        let (_x_n1, x) = x.split_at(n + 1);
        let (x_m, x) = x.split_at(m);
        let (x_p, _) = x.split_at(p);

        let f1 = F::one();

        // y = a*symvec_p^T*x_sn + 0*x_n1 + a*vec_h^T*x_m + a*vec_b^T*x_p + b*y
        self.symvec_p.trans_op(alpha, x_sn, beta, y);
        self.vec_h.trans_op(alpha, x_m, f1, y);
        self.vec_b.trans_op(alpha, x_p, f1, y);
    }
}

//

pub struct ProbQPCone<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    n: usize,
    m: usize,
    p: usize,
    cone_psd: ConePSD<'a, L, F>,
    cone_rpos: ConeRPos<F>,
    cone_zero: ConeZero<F>,
}

impl<'a, L, F> Cone<F> for ProbQPCone<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn proj(&mut self, eps_zero: F, x: &mut[F]) -> Result<(), SolverError>
    {
        let (n, m, p) = (self.n, self.m, self.p);
        let sn = n * (n + 1) / 2;
        let (x_s, x) = x.split_at_mut(sn + n + 1);
        let (x_m, x) = x.split_at_mut(m);
        let (x_p, _) = x.split_at_mut(p);

        self.cone_psd.proj(eps_zero, x_s)?;
        self.cone_rpos.proj(eps_zero, x_m)?;
        self.cone_zero.proj(eps_zero, x_p)?;
        Ok(())
    }

    fn dual_proj(&mut self, eps_zero: F, x: &mut[F]) -> Result<(), SolverError>
    {
        let (n, m, p) = (self.n, self.m, self.p);
        let sn = n * (n + 1) / 2;
        let (x_s, x) = x.split_at_mut(sn + n + 1);
        let (x_m, x) = x.split_at_mut(m);
        let (x_p, _) = x.split_at_mut(p);

        self.cone_psd.dual_proj(eps_zero, x_s)?;
        self.cone_rpos.dual_proj(eps_zero, x_m)?;
        self.cone_zero.dual_proj(eps_zero, x_p)?;
        Ok(())
    }
}

//

pub struct ProbQP<L, F>
where L: LinAlgEx<F>, F: Float
{
    sym_p: MatBuild<L, F>,
    vec_q: MatBuild<L, F>,
    mat_g: MatBuild<L, F>,
    vec_h: MatBuild<L, F>,
    mat_a: MatBuild<L, F>,
    vec_b: MatBuild<L, F>,

    symvec_p: MatBuild<L, F>,

    w_cone_psd: Vec<F>,
    w_solver: Vec<F>,
}

impl<L, F> ProbQP<L, F>
where L: LinAlgEx<F>, F: Float
{
    pub fn new(
        sym_p: MatBuild<L, F>, vec_q: MatBuild<L, F>,
        mat_g: MatBuild<L, F>, vec_h: MatBuild<L, F>,
        mat_a: MatBuild<L, F>, vec_b: MatBuild<L, F>) -> Self
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

        let f1 = F::one();
        let f2 = f1 + f1;
        let fsqrt2 = f2.sqrt();
    
        let symvec_p = sym_p.clone()
                       .scale_nondiag(fsqrt2)
                       .reshape_colvec();

        ProbQP {
            sym_p,
            vec_q,
            mat_g,
            vec_h,
            mat_a,
            vec_b,
            symvec_p,
            w_cone_psd: Vec::new(),
            w_solver: Vec::new(),
        }
    }

    pub fn problem(&mut self) -> (ProbQPOpC<L, F>, ProbQPOpA<L, F>, ProbQPOpB<L, F>, ProbQPCone<'_, L, F>, &mut[F])
    {
        let n = self.vec_q.size().0;
        let m = self.vec_h.size().0;
        let p = self.vec_b.size().0;
        let sn = n * (n + 1) / 2;

        let f0 = F::zero();

        let op_c = ProbQPOpC {
            vec_q: &self.vec_q,
        };
        let op_a = ProbQPOpA {
            sym_p: &self.sym_p,
            mat_g: &self.mat_g,
            mat_a: &self.mat_a,
        };
        let op_b = ProbQPOpB {
            n,
            symvec_p: &self.symvec_p,
            vec_h: &self.vec_h,
            vec_b: &self.vec_b,
        };

        self.w_cone_psd.resize(ConePSD::<L, _>::query_worklen(sn + n + 1), f0);
        let cone = ProbQPCone {
            n, m, p,
            cone_psd: ConePSD::new(self.w_cone_psd.as_mut()),
            cone_rpos: ConeRPos::new(),
            cone_zero: ConeZero::new(),
        };

        self.w_solver.resize(Solver::<L, _>::query_worklen(op_a.size()), f0);

        (op_c, op_a, op_b, cone, self.w_solver.as_mut())
    }
}

//

#[cfg(test)]
extern crate intel_mkl_src;

#[test]
fn test_qp1()
{
    use float_eq::assert_float_eq;
    use crate::logger::PrintLogger;
    use crate::operator::MatType;
    use crate::linalg::F64LAPACK;
    use crate::linalg::FloatGeneric;
    
    type _LA = F64LAPACK;
    type LA = FloatGeneric<f64>;
    type ASolver = Solver<LA, f64>;
    type AProbQP = ProbQP<LA, f64>;
    type AMatBuild = MatBuild<LA, f64>;

    let n = 2; // x0, x1
    let m = 1;
    let p = 0;
    
    // (1/2)(x - a)^2 + const
    let mut sym_p = AMatBuild::new(MatType::SymPack(n));
    sym_p[(0, 0)] = 1.;
    sym_p[(1, 1)] = 1.;

    let mut vec_q = AMatBuild::new(MatType::General(n, 1));
    vec_q[(0, 0)] = -(-1.); // -a0
    vec_q[(1, 0)] = -(-2.); // -a1
    
    // 1 - x0/b0 - x1/b1 <= 0
    let mut mat_g = AMatBuild::new(MatType::General(m, n));
    mat_g[(0, 0)] = -1. / 2.; // -1/b0
    mat_g[(0, 1)] = -1. / 3.; // -1/b1
    
    let mut vec_h = AMatBuild::new(MatType::General(m, 1));
    vec_h[(0, 0)] = -1.;

    let mat_a = AMatBuild::new(MatType::General(p, n));

    let vec_b = AMatBuild::new(MatType::General(p, 1));

    let s = ASolver::new();
    println!("{:?}", s.par);
    let mut qp = AProbQP::new(sym_p, vec_q, mat_g, vec_h, mat_a, vec_b);
    let rslt = s.solve(qp.problem(), PrintLogger).unwrap();
    println!("{:?}", rslt);

    assert_float_eq!(rslt.0[0..2], [2., 0.].as_ref(), abs_all <= 1e-3);
}
    