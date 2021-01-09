use crate::matop::{MatOp, MatType};
use crate::matbuild::MatBuild;
use crate::solver::{Operator, Cone, LinAlg, SolverParam, SolverError, solver_query_worklen};
use crate::linalg::F64BLAS;
use crate::cone::{ConePSD, ConeRPos, ConeZero};

//

pub struct ProbQPOpC<'a>
{
    vec_q: MatOp<'a>,
}

impl<'a> ProbQPOpC<'a>
{
    fn dim(&self) -> usize
    {
        let (n, one) = self.vec_q.size();
        assert_eq!(one, 1);
        n
    }
}

impl<'a> Operator<f64> for ProbQPOpC<'a>
{
    fn size(&self) -> (usize, usize)
    {
        let n = self.dim();

        (n + 1, 1)
    }

    fn op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        let n = self.dim();
        let (y_n, y_1) = y.split_at_mut(n);

        // y_n = a*vec_q*x + b*y_n;
        self.vec_q.op(alpha, x, beta, y_n);

        // y_1 = a*1*x + b*y_1;
        F64BLAS::scale(beta, y_1);
        F64BLAS::add(alpha, x, y_1);
    }

    fn trans_op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        let n = self.dim();
        let (x_n, x_1) = x.split_at(n);

        // y = a*vec_q^T*x_n + a*1*x_1 + b*y;
        self.vec_q.trans_op(alpha, x_n, beta, y);
        F64BLAS::add(alpha, x_1, y);
    }
}

//

pub struct ProbQPOpA<'a>
{
    sym_p: MatOp<'a>,
    mat_g: MatOp<'a>,
    mat_a: MatOp<'a>,
}

impl<'a> ProbQPOpA<'a>
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

impl<'a> Operator<f64> for ProbQPOpA<'a>
{
    fn size(&self) -> (usize, usize)
    {
        let (n, sn, m, p) = self.dim();

        ((sn + n + 1) + m + p, n + 1)
    }

    fn op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        let (n, sn, m, p) = self.dim();
        let (x_n, x) = x.split_at(n);
        let (x_1, _) = x.split_at(1);
        let (y_sn, y) = y.split_at_mut(sn);
        let (y_n, y) = y.split_at_mut(n);
        let (y_1, y) = y.split_at_mut(1);
        let (y_m, y) = y.split_at_mut(m);
        let (y_p, _) = y.split_at_mut(p);

        // y_sn = 0*x_n + 0*x_1 + b*y_sn
        F64BLAS::scale(beta, y_sn);

        // y_n = a*-sqrt(2)*sym_p*x_n + b*y_n
        self.sym_p.op(-alpha * 2_f64.sqrt(), x_n, beta, y_n);

        // y_1 = 0*x_n + a*-2*x_1 + b*y_1
        F64BLAS::scale(beta, y_1);
        F64BLAS::add(-2. * alpha, x_1, y_1);

        // y_m = a*mat_g*x_n + 0*x_1 + b*y_m
        self.mat_g.op(alpha, x_n, beta, y_m);

        // y_p = a*mat_a*x_n + 0*x_1 + b*y_p
        self.mat_a.op(alpha, x_n, beta, y_p);
    }

    fn trans_op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        let (n, sn, m, p) = self.dim();
        let (_x_sn, x) = x.split_at(sn);
        let (x_n, x) = x.split_at(n);
        let (x_1, x) = x.split_at(1);
        let (x_m, x) = x.split_at(m);
        let (x_p, _) = x.split_at(p);
        let (y_n, y) = y.split_at_mut(n);
        let (y_1, _) = y.split_at_mut(1);

        // y_n = 0*x_sn + a*-sqrt(2)*sym_p*x_n + 0*x_1 + a*mat_g^T*x_m + a*mat_a^T*x_p + b*y_n
        self.sym_p.trans_op(-alpha * 2_f64.sqrt(), x_n, beta, y_n);
        self.mat_g.trans_op(alpha, x_m, 1., y_n);
        self.mat_a.trans_op(alpha, x_p, 1., y_n);

        // y_1 = 0*x_sn + 0*x_n + a*-2*x_1 + 0*x_m + 0*x_p + b*y_1
        F64BLAS::scale(beta, y_1);
        F64BLAS::add(-2. * alpha, x_1, y_1);
    }
}

//

pub struct ProbQPOpB<'a>
{
    n: usize,
    symvec_p: MatOp<'a>,
    vec_h: MatOp<'a>,
    vec_b: MatOp<'a>,
}

impl<'a> ProbQPOpB<'a>
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

impl<'a> Operator<f64> for ProbQPOpB<'a>
{
    fn size(&self) -> (usize, usize)
    {
        let (n, sn, m, p) = self.dim();

        ((sn + n + 1) + m + p, 1)
    }

    fn op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        let (n, sn, m, p) = self.dim();
        let (y_sn, y) = y.split_at_mut(sn);
        let (y_n1, y) = y.split_at_mut(n + 1);
        let (y_m, y) = y.split_at_mut(m);
        let (y_p, _) = y.split_at_mut(p);

        // y_sn = a*symvec_p*x + b*y_sn
        self.symvec_p.op(alpha, x, beta, y_sn);

        // y_n1 = 0*x + b*y_n1
        F64BLAS::scale(beta, y_n1);

        // y_m = a*vec_h*x + b*y_m
        self.vec_h.op(alpha, x, beta, y_m);

        // y_p = a*vec_b*x + b*y_p
        self.vec_b.op(alpha, x, beta, y_p);
    }

    fn trans_op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        let (n, sn, m, p) = self.dim();
        let (x_sn, x) = x.split_at(sn);
        let (_x_n1, x) = x.split_at(n + 1);
        let (x_m, x) = x.split_at(m);
        let (x_p, _) = x.split_at(p);

        // y = a*symvec_p^T*x_sn + 0*x_n1 + a*vec_h^T*x_m + a*vec_b^T*x_p + b*y
        self.symvec_p.trans_op(alpha, x_sn, beta, y);
        self.vec_h.trans_op(alpha, x_m, 1., y);
        self.vec_b.trans_op(alpha, x_p, 1., y);
    }
}

//

pub struct ProbQPCone<'a>
{
    n: usize,
    m: usize,
    p: usize,
    cone_psd: ConePSD<'a>,
    cone_rpos: ConeRPos,
    cone_zero: ConeZero,
}

impl<'a> Cone<f64> for ProbQPCone<'a>
{
    fn proj(&mut self, par: &SolverParam<f64>, x: &mut[f64]) -> Result<(), SolverError>
    {
        let (n, m, p) = (self.n, self.m, self.p);
        let sn = n * (n + 1) / 2;
        let (x_s, x) = x.split_at_mut(sn + n + 1);
        let (x_m, x) = x.split_at_mut(m);
        let (x_p, _) = x.split_at_mut(p);

        self.cone_psd.proj(par, x_s)?;
        self.cone_rpos.proj(par, x_m)?;
        self.cone_zero.proj(par, x_p)?;
        Ok(())
    }

    fn dual_proj(&mut self, par: &SolverParam<f64>, x: &mut[f64]) -> Result<(), SolverError>
    {
        let (n, m, p) = (self.n, self.m, self.p);
        let sn = n * (n + 1) / 2;
        let (x_s, x) = x.split_at_mut(sn + n + 1);
        let (x_m, x) = x.split_at_mut(m);
        let (x_p, _) = x.split_at_mut(p);

        self.cone_psd.dual_proj(par, x_s)?;
        self.cone_rpos.dual_proj(par, x_m)?;
        self.cone_zero.dual_proj(par, x_p)?;
        Ok(())
    }
}

//

pub struct ProbQP
{
    sym_p: MatBuild,
    vec_q: MatBuild,
    mat_g: MatBuild,
    vec_h: MatBuild,
    mat_a: MatBuild,
    vec_b: MatBuild,

    symvec_p: MatBuild,

    w_cone_psd: Vec<f64>,
    w_solver: Vec<f64>,
}

impl ProbQP
{
    pub fn new(
        sym_p: MatBuild, vec_q: MatBuild,
        mat_g: MatBuild, vec_h: MatBuild,
        mat_a: MatBuild, vec_b: MatBuild) -> Option<Self>
    {
        let n = vec_q.typ().size().0;
        let m = vec_h.typ().size().0;
        let p = vec_b.typ().size().0;
    
        // TODO: error
        assert_eq!(sym_p.typ(), &MatType::SymPack(n));
        assert_eq!(vec_q.typ().size(), (n, 1));
        assert_eq!(mat_g.typ().size(), (m, n));
        assert_eq!(vec_h.typ().size(), (m, 1));
        assert_eq!(mat_a.typ().size(), (p, n));
        assert_eq!(vec_b.typ().size(), (p, 1));
    
        let symvec_p = sym_p.clone()
                       .scale_nondiag(2_f64.sqrt())
                       .reshape_colvec();

        Some(ProbQP {
            sym_p,
            vec_q,
            mat_g,
            vec_h,
            mat_a,
            vec_b,
            symvec_p,
            w_cone_psd: Vec::new(),
            w_solver: Vec::new(),
        })
    }

    pub fn problem(&mut self) -> (ProbQPOpC, ProbQPOpA, ProbQPOpB, ProbQPCone, &mut[f64])
    {
        let n = self.vec_q.typ().size().0;
        let m = self.vec_h.typ().size().0;
        let p = self.vec_b.typ().size().0;
        let sn = n * (n + 1) / 2;

        let op_c = ProbQPOpC {
            vec_q: MatOp::from(&self.vec_q),
        };
        let op_a = ProbQPOpA {
            sym_p: MatOp::from(&self.sym_p),
            mat_g: MatOp::from(&self.mat_g),
            mat_a: MatOp::from(&self.mat_a),
        };
        let op_b = ProbQPOpB {
            n,
            symvec_p: MatOp::from(&self.symvec_p),
            vec_h: MatOp::from(&self.vec_h),
            vec_b: MatOp::from(&self.vec_b),
        };

        self.w_cone_psd.resize(ConePSD::query_worklen(sn + n + 1), 0.);
        let cone = ProbQPCone {
            n, m, p,
            cone_psd: ConePSD::new(self.w_cone_psd.as_mut()),
            cone_rpos: ConeRPos,
            cone_zero: ConeZero,
        };

        self.w_solver.resize(solver_query_worklen(op_a.size()), 0.);

        (op_c, op_a, op_b, cone, self.w_solver.as_mut())
    }
}


#[test]
fn test_qp1() {
    use crate::solver::Solver;
    use crate::logger::*;
    use float_eq::assert_float_eq;

    const N: usize = 2; // x0, x1
    const M: usize = 1;
    const P: usize = 0;
    
    // (1/2)(x - a)^2 + const
    let mut sym_p = MatBuild::new(MatType::SymPack(N));
    sym_p[(0, 0)] = 1.;
    sym_p[(1, 1)] = 1.;

    let mut vec_q = MatBuild::new(MatType::General(N, 1));
    vec_q[(0, 0)] = -(-1.); // -a0
    vec_q[(1, 0)] = -(-2.); // -a1
    
    // 1 - x0/b0 - x1/b1 <= 0
    let mut mat_g = MatBuild::new(MatType::General(M, N));
    mat_g[(0, 0)] = -1. / 2.; // -1/b0
    mat_g[(0, 1)] = -1. / 3.; // -1/b1
    
    let mut vec_h = MatBuild::new(MatType::General(M, 1));
    vec_h[(0, 0)] = -1.;

    let mat_a = MatBuild::new(MatType::General(P, N));

    let vec_b = MatBuild::new(MatType::General(P, 1));

    //let mut stdout = std::io::stdout();
    //let log = IoLogger(&mut stdout);
    let log = NullLogger;

    let s = Solver::new(F64BLAS, log);
    println!("{:?}", s.par);
    let mut qp = ProbQP::new(sym_p, vec_q, mat_g, vec_h, mat_a, vec_b).unwrap();
    let rslt = s.solve(qp.problem()).unwrap();
    println!("{:?}", rslt);

    assert_float_eq!(rslt.0[0..2], [2., 0.].as_ref(), abs_all <= 1e-3);
}
    