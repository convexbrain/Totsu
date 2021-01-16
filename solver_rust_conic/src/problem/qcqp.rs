use num::Float;
use core::marker::PhantomData;
use crate::solver::{SolverError, Solver};
use crate::linalg::LinAlgEx;
use crate::operator::{Operator, MatBuild};
use crate::cone::{Cone, ConePSD, ConeZero};

//

pub struct ProbQCQPOpC<L, F>
where L: LinAlgEx<F>, F: Float
{
    ph_l: PhantomData<L>,
    ph_f: PhantomData<F>,
    n: usize,
}

impl<L, F> Operator<F> for ProbQCQPOpC<L, F>
where L: LinAlgEx<F>, F: Float
{
    fn size(&self) -> (usize, usize)
    {
        (self.n + 1, 1)
    }

    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let n = self.n;
        let (y_n, y_1) = y.split_at_mut(n);

        // y_n = a*0*x + b*y_n;
        L::scale(beta, y_n);

        // y_1 = a*1*x + b*y_1;
        L::scale(beta, y_1);
        L::add(alpha, x, y_1);
    }

    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let n = self.n;
        let (_x_n, x_1) = x.split_at(n);

        // y = a*0*x_n + a*1*x_1 + b*y;
        L::scale(beta, y);
        L::add(alpha, x_1, y);
    }
}

//

pub struct ProbQCQPOpA<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    syms_p: &'a[MatBuild<L, F>],
    vecs_q: &'a[MatBuild<L, F>],
    mat_a: &'a MatBuild<L, F>,
}

impl<'a, L, F> ProbQCQPOpA<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn dim(&self) -> (usize, usize, usize, usize)
    {
        let m1 = self.syms_p.len();
        let (p, n) = self.mat_a.size();
        let sn = n * (n + 1) / 2;

        (n, sn, m1, p)
    }
}

impl<'a, L, F> Operator<F> for ProbQCQPOpA<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn size(&self) -> (usize, usize)
    {
        let (n, sn, m1, p) = self.dim();

        (m1 * (sn + (n + 1) + n + 1 + 1) + p, n + 1)
    }

    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (n, sn, _m1, _p) = self.dim();

        let (x_n, x) = x.split_at(n);
        let (x_1, _) = x.split_at(1);

        let f2 = F::one() + F::one();
        let fsqrt2 = f2.sqrt();

        let mut i = 0;
        let mut spl_y = y;
        for (sym_p, vec_q) in self.syms_p.iter().zip(self.vecs_q) {
            let (y_sn_n1, spl) = spl_y.split_at_mut(sn + (n + 1));
            let (y_n, spl) = spl.split_at_mut(n);
            let (y_1, spl) = spl.split_at_mut(1);
            let (y_o1, spl) = spl.split_at_mut(1);
            spl_y = spl;

            // y_sn_n1 = 0*x_n + 0*x_1 + b*y_sn_n1
            L::scale(beta, y_sn_n1);

            // y_n = a*-sqrt(2)*sym_p*x_n + 0*x_1 + b*y_n
            sym_p.op(-alpha * fsqrt2, x_n, beta, y_n);

            // y_1 = 0*x_n + 0*x_1 + b*y_1
            L::scale(beta, y_1);

            // y_o1 = a*2*vec_q^T*x_n + a*-2*x_1 + b*y_o1  (i = 0)
            //      = a*2*vec_q^T*x_n +    0*x_1 + b*y_o1  (i > 0)
            vec_q.trans_op(f2 * alpha, x_n, beta, y_o1);
            if i == 0 {
                L::add(-f2 * alpha, x_1, y_o1)
            }

            i += 1;
        }

        let y_p = spl_y;

        // y_p = a*mat_a*x_n + 0*x_1 + b*y_p
        self.mat_a.op(alpha, x_n, beta, y_p);
    }

    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (n, sn, _m1, _p) = self.dim();

        let (y_n, y) = y.split_at_mut(n);
        let (y_1, _) = y.split_at_mut(1);

        let f1 = F::one();
        let f2 = f1 + f1;
        let fsqrt2 = f2.sqrt();

        // y_n = b*y_n + ...
        // y_1 = b*y_1 + ...
        L::scale(beta, y_n);
        L::scale(beta, y_1);

        let mut i = 0;
        let mut spl_x = x;
        for (sym_p, vec_q) in self.syms_p.iter().zip(self.vecs_q) {
            let (_x_sn_n1, spl) = spl_x.split_at(sn + (n + 1));
            let (x_n, spl) = spl.split_at(n);
            let (_x_1, spl) = spl.split_at(1);
            let (x_o1, spl) = spl.split_at(1);
            spl_x = spl;

            // y_n = ... + 0*x_sn_n1 + ...
            // y_1 = ... + 0*x_sn_n1 + ...

            // y_n = ... + a*-sqrt(2)*sym_p*x_n + ...
            // y_1 = ... + 0*x_n + ...
            sym_p.op(-alpha * fsqrt2, x_n, f1, y_n);

            // y_n = ... + 0*x_1 + ...
            // y_1 = ... + 0*x_1 + ...

            // y_n = ... + a*2*vec_q*x_o1 + ...
            // y_1 = ... + a*-2*x_o1 + ...  (i = 0)
            //       ... +    0*x_o1 + ...  (i > 0)
            vec_q.op(f2 * alpha, x_o1, f1, y_n);
            if i == 0 {
                L::add(-f2 * alpha, x_o1, y_1);
            }

            i += 1;
        }

        let x_p = spl_x;

        // y_n = ... + a*mat_a^T*x_p
        // y_1 = ... +         0*x_p
        self.mat_a.trans_op(alpha, x_p, beta, y_n);
    }
}

//

pub struct ProbQCQPOpB<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    n: usize,
    symvecs_p: &'a[MatBuild<L, F>],
    scls_r: &'a[F],
    vec_b: &'a MatBuild<L, F>,
}

impl<'a, L, F> ProbQCQPOpB<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn dim(&self) -> (usize, usize, usize, usize)
    {
        let n = self.n;
        let m1 = self.symvecs_p.len();
        let (p, one) = self.vec_b.size();
        assert_eq!(one, 1);
        let sn = n * (n + 1) / 2;

        (n, sn, m1, p)
    }
}

impl<'a, L, F> Operator<F> for ProbQCQPOpB<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn size(&self) -> (usize, usize)
    {
        let (n, sn, m1, p) = self.dim();

        (m1 * (sn + (n + 1) + n + 1 + 1) + p, 1)
    }

    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (n, sn, _m1, _p) = self.dim();

        let f1 = F::one();
        let f2 = f1 + f1;

        let mut spl_y = y;
        for (symvec_p, scl_r) in self.symvecs_p.iter().zip(self.scls_r) {
            let (y_sn, spl) = spl_y.split_at_mut(sn);
            let (y_n1_n_1, spl) = spl.split_at_mut((n + 1) + n + 1);
            let (y_o1, spl) = spl.split_at_mut(1);
            spl_y = spl;

            // y_sn = alpha*symvec_p*x + b*y_sn
            symvec_p.op(alpha, x, beta, y_sn);

            // y_n1_n_1 = 0*x + b*y_n1_n_1
            L::scale(beta, y_n1_n_1);

            // y_o1 = a*-2*scl_r*x + b*y_o1
            L::scale(beta, y_o1);
            L::add(-f2 * alpha * *scl_r, x, y_o1);
        }

        let y_p = spl_y;

        // y_p = a*vec_b*x + b*y_p
        self.vec_b.op(alpha, x, beta, y_p);
    }

    fn trans_op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let (n, sn, _m1, _p) = self.dim();

        let f1 = F::one();
        let f2 = f1 + f1;

        // y = b*y + ...
        L::scale(beta, y);

        let mut spl_x = x;
        for (symvec_p, scl_r) in self.symvecs_p.iter().zip(self.scls_r) {
            let (x_sn, spl) = spl_x.split_at(sn);
            let (_x_n1_n_1, spl) = spl.split_at((n + 1) + n + 1);
            let (x_o1, spl) = spl.split_at(1);
            spl_x = spl;

            // y = ... + a*symvec_p^T*x_sn + ...
            symvec_p.trans_op(alpha, x_sn, f1, y);

            // y = ... + 0*x_n1_n_1 + ...

            // y = ... + a*-2*scl_r*x_o1 + ...
            L::add(-f2 * alpha * *scl_r, x_o1, y);
        }

        let x_p = spl_x;

        // y = ... + a*vec_b^T*x_p
        self.vec_b.trans_op(alpha, x_p, f1, y);
    }
}

//

pub struct ProbQCQPCone<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    n: usize,
    m1: usize,
    cone_psd: ConePSD<'a, L, F>,
    cone_zero: ConeZero<F>,
}

impl<'a, L, F> Cone<F> for ProbQCQPCone<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn proj(&mut self, eps_zero: F, x: &mut[F]) -> Result<(), SolverError>
    {
        let n = self.n;
        let m1 = self.m1;
        let sn = n * (n + 1) / 2;

        let mut spl_x = x;
        for _ in 0.. m1 {
            let (x_sn_n1_n_1_1, spl) = spl_x.split_at_mut(sn + (n + 1) + n + 1 + 1);
            spl_x = spl;

            self.cone_psd.proj(eps_zero, x_sn_n1_n_1_1)?;
        }

        let x_p = spl_x;

        self.cone_zero.proj(eps_zero, x_p)?;
        Ok(())
    }

    fn dual_proj(&mut self, eps_zero: F, x: &mut[F]) -> Result<(), SolverError>
    {
        let n = self.n;
        let m1 = self.m1;
        let sn = n * (n + 1) / 2;

        let mut spl_x = x;
        for _ in 0.. m1 {
            let (x_sn_n1_n_1_1, spl) = spl_x.split_at_mut(sn + (n + 1) + n + 1 + 1);
            spl_x = spl;

            self.cone_psd.dual_proj(eps_zero, x_sn_n1_n_1_1)?;
        }

        let x_p = spl_x;

        self.cone_zero.dual_proj(eps_zero, x_p)?;
        Ok(())
    }
}

//

pub struct ProbQCQP<L, F>
where L: LinAlgEx<F>, F: Float
{
    syms_p: Vec<MatBuild<L, F>>,
    vecs_q: Vec<MatBuild<L, F>>,
    scls_r: Vec<F>,
    mat_a: MatBuild<L, F>,
    vec_b: MatBuild<L, F>,

    symvecs_p: Vec<MatBuild<L, F>>,

    w_cone_psd: Vec<F>,
    w_solver: Vec<F>,
}

impl<L, F> ProbQCQP<L, F>
where L: LinAlgEx<F>, F: Float
{
    pub fn new(
        syms_p: Vec<MatBuild<L, F>>, vecs_q: Vec<MatBuild<L, F>>, scls_r: Vec<F>,
        mat_a: MatBuild<L, F>, vec_b: MatBuild<L, F>) -> Self
    {
        let (p, n) = mat_a.size();
        let m1 = syms_p.len();
    
        assert_eq!(syms_p.len(), m1);
        assert_eq!(vecs_q.len(), m1);
        assert_eq!(scls_r.len(), m1);
        for i in 0.. m1 {
            assert!(syms_p[i].is_sympack());
            assert_eq!(syms_p[i].size(), (n, n));
            assert_eq!(vecs_q[i].size(), (n, 1));
        }
        assert_eq!(mat_a.size(), (p, n));
        assert_eq!(vec_b.size(), (p, 1));

        let f1 = F::one();
        let f2 = f1 + f1;
        let fsqrt2 = f2.sqrt();
    
        let mut symvecs_p = Vec::new();
        for i in 0.. m1 {
            symvecs_p.push(syms_p[i].clone()
                           .scale_nondiag(fsqrt2)
                           .reshape_colvec());
        }

        ProbQCQP {
            syms_p,
            vecs_q,
            scls_r,
            mat_a,
            vec_b,
            symvecs_p,
            w_cone_psd: Vec::new(),
            w_solver: Vec::new(),
        }
    }
    
    pub fn problem(&mut self) -> (ProbQCQPOpC<L, F>, ProbQCQPOpA<L, F>, ProbQCQPOpB<L, F>, ProbQCQPCone<'_, L, F>, &mut[F])
    {
        let n = self.mat_a.size().1;
        let m1 = self.syms_p.len();
        let sn = n * (n + 1) / 2;
    
        let f0 = F::zero();

        let op_c = ProbQCQPOpC {
            ph_l: PhantomData,
            ph_f: PhantomData,
            n,
        };
        let op_a = ProbQCQPOpA {
            syms_p: &self.syms_p,
            vecs_q: &self.vecs_q,
            mat_a: &self.mat_a,
        };
        let op_b = ProbQCQPOpB {
            n,
            symvecs_p: &self.symvecs_p,
            scls_r: &self.scls_r,
            vec_b: &self.vec_b,
        };

        self.w_cone_psd.resize(ConePSD::<L, _>::query_worklen(sn + (n + 1) + n + 1 + 1), f0);
        let cone = ProbQCQPCone {
            n,
            m1,
            cone_psd: ConePSD::new(self.w_cone_psd.as_mut()),
            cone_zero: ConeZero::new(),
        };

        self.w_solver.resize(Solver::<L, _>::query_worklen(op_a.size()), f0);

        (op_c, op_a, op_b, cone, self.w_solver.as_mut())
    }
}
