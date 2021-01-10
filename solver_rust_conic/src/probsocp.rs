use crate::matop::{MatOp, MatType};
use crate::matbuild::MatBuild;
use crate::solver::{Operator, Cone, SolverError, Solver};
use crate::linalgex::LinAlgEx;
use crate::cone::{ConeSOC, ConeZero};
use core::marker::PhantomData;
use num::Float;

//

pub struct ProbSOCPOpC<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    _ph_l: PhantomData<L>,
    vec_f: MatOp<'a, L, F>,
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
    _ph_l: PhantomData<L>,
    sli_mat_g_vec_c: &'a[(MatOp<'a, L, F>, MatOp<'a, L, F>)],
    mat_a: MatOp<'a, L, F>,
}

impl<'a, L, F> Operator<F> for ProbSOCPOpA<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn size(&self) -> (usize, usize)
    {
        let (p, n) = self.mat_a.size();

        let mut ni1_sum = 0;

        for (mat_g, vec_c) in self.sli_mat_g_vec_c {
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

        for (mat_g, vec_c) in self.sli_mat_g_vec_c {
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

        for (mat_g, vec_c) in self.sli_mat_g_vec_c {
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
    _ph_l: PhantomData<L>,
    sli_vec_h_scl_d: &'a[(MatOp<'a, L, F>, F)],
    vec_b: MatOp<'a, L, F>,
}

impl<'a, L, F> Operator<F> for ProbSOCPOpB<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn size(&self) -> (usize, usize)
    {
        let (p, one) = self.vec_b.size();
        assert_eq!(one, 1);

        let mut ni1_sum = 0;

        for (vec_h, _) in self.sli_vec_h_scl_d {
            let (ni, one) = vec_h.size();
            assert_eq!(one, 1);

            ni1_sum += ni + 1;
        }

        (ni1_sum + p, 1)
    }

    fn op(&self, alpha: F, x: &[F], beta: F, y: &mut[F])
    {
        let mut spl_y = y;

        for (vec_h, scl_d) in self.sli_vec_h_scl_d {
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

        for (vec_h, scl_d) in self.sli_vec_h_scl_d {
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
    sli_ni: &'a[usize],
    cone_soc: ConeSOC<L, F>,
    cone_zero: ConeZero<F>,
}

impl<'a, L, F> Cone<F> for ProbSOCPCone<'a, L, F>
where L: LinAlgEx<F>, F: Float
{
    fn proj(&mut self, eps_zero: F, x: &mut[F]) -> Result<(), SolverError>
    {
        let mut spl_x = x;

        for ni in self.sli_ni {
            let (x_ni1, spl) = spl_x.split_at_mut(ni + 1);
            spl_x = spl;

            self.cone_soc.proj(eps_zero, x_ni1)?;
        }

        let x_p = spl_x;

        self.cone_zero.proj(eps_zero, x_p)?;
        Ok(())
    }

    fn dual_proj(&mut self, eps_zero: F, x: &mut[F]) -> Result<(), SolverError>
    {
        let mut spl_x = x;

        for ni in self.sli_ni {
            let (x_ni1, spl) = spl_x.split_at_mut(ni + 1);
            spl_x = spl;

            self.cone_soc.dual_proj(eps_zero, x_ni1)?;
        }

        let x_p = spl_x;

        self.cone_zero.dual_proj(eps_zero, x_p)?;
        Ok(())
    }
}

//

// TODO
