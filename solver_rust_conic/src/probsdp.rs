use crate::matop::MatType;
use crate::matbuild::MatBuild;
use crate::solver::{Operator, Cone, SolverError, Solver};
use crate::linalgex::LinAlgEx;
use crate::cone::{ConePSD, ConeZero};
use num::Float;

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

        let (y_sk, y) = y.split_at_mut(sk);
        let (y_p, _) = y.split_at_mut(p);

        // y_sk = a*symmat_f*x + b*y_sk
        self.symmat_f.op(alpha, x, beta, y_sk);

        // y_p = a*mat_a*x + b*y_p
        self.mat_a.op(alpha, x, beta, y_p);
    }

    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (_n, sk, p) = self.dim();

        let (x_sk, x) = x.split_at(sk);
        let (x_p, _) = x.split_at(p);

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

        let (y_sk, y) = y.split_at_mut(sk);
        let (y_p, _) = y.split_at_mut(p);

        // y_sk = a*-symmat_f*x + b*y_sk
        self.symvec_f_n.op(-alpha, x, beta, y_sk);

        // y_p = a*vec_b*x + b*y_p
        self.vec_b.op(alpha, x, beta, y_p);
    }

    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (_n, sk, p) = self.dim();

        let (x_sk, x) = x.split_at(sk);
        let (x_p, _) = x.split_at(p);

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
    fn proj(&mut self, eps_zero: F, x: &mut[F]) -> Result<(), SolverError>
    {
        let (sk, p) = (self.sk, self.p);
        let (x_sk, x) = x.split_at_mut(sk);
        let (x_p, _) = x.split_at_mut(p);

        self.cone_psd.proj(eps_zero, x_sk)?;
        self.cone_zero.proj(eps_zero, x_p)?;
        Ok(())
    }

    fn dual_proj(&mut self, eps_zero: F, x: &mut[F]) -> Result<(), SolverError>
    {
        let (sk, p) = (self.sk, self.p);
        let (x_sk, x) = x.split_at_mut(sk);
        let (x_p, _) = x.split_at_mut(p);

        self.cone_psd.dual_proj(eps_zero, x_sk)?;
        self.cone_zero.dual_proj(eps_zero, x_p)?;
        Ok(())
    }
}

//

pub struct ProbSDP<L, F>
where L: LinAlgEx<F>, F: Float
{
    vec_c: MatBuild<L, F>,
    mat_a: MatBuild<L, F>,
    vec_b: MatBuild<L, F>,

    symmat_f: MatBuild<L, F>,
    symvec_f_n: MatBuild<L, F>,

    w_cone_psd: Vec<F>,
    w_solver: Vec<F>,
}

impl<L, F> ProbSDP<L, F>
where L: LinAlgEx<F>, F: Float
{
    pub fn new(
        vec_c: MatBuild<L, F>,
        mut syms_f: Vec<MatBuild<L, F>>,
        mat_a: MatBuild<L, F>, vec_b: MatBuild<L, F>) -> Self
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
            w_cone_psd: Vec::new(),
            w_solver: Vec::new(),
        }
    }

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
            cone_psd: ConePSD::new(self.w_cone_psd.as_mut()),
            cone_zero: ConeZero::new(),
        };

        self.w_solver.resize(Solver::<L, _>::query_worklen(op_a.size()), f0);

        (op_c, op_a, op_b, cone, self.w_solver.as_mut())
    }
}

//

#[test]
fn test_sdp1()
{
    use float_eq::assert_float_eq;
    use crate::stdlogger::PrintLogger;
    use crate::matop::MatType;
    use crate::f64lapack::F64LAPACK;
    
    type ASolver = Solver<F64LAPACK, f64>;
    type AProbSDP = ProbSDP<F64LAPACK, f64>;
    type AMatBuild = MatBuild<F64LAPACK, f64>;

    let n = 2;
    let p = 0;
    let k = 2;

    let vec_c = AMatBuild::new(MatType::General(n, 1)).iter_colmaj(&[
        1., 1.
    ]);

    let mut syms_f = vec![AMatBuild::new(MatType::SymPack(k)); n + 1];

    syms_f[0].set_iter_rowmaj(&[
        -1., 0.,
         0., 0.
    ]);
    syms_f[1].set_iter_rowmaj(&[
        0.,  0.,
        0., -1.
    ]);
    syms_f[2].set_iter_rowmaj(&[
        3., 0.,
        0., 4.
    ]);

    let mat_a = AMatBuild::new(MatType::General(p, n));

    let vec_b = AMatBuild::new(MatType::General(p, 1));


    let s = ASolver::new();
    println!("{:?}", s.par);
    let mut sdp = AProbSDP::new(vec_c, syms_f, mat_a, vec_b);
    let rslt = s.solve(sdp.problem(), PrintLogger).unwrap();
    println!("{:?}", rslt);

    assert_float_eq!(rslt.0, [3., 4.].as_ref(), abs_all <= 1e-3);
}
