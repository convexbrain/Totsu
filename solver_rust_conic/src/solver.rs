
fn normalize(x: &mut[f64])
{
    let norm = unsafe { cblas::dnrm2(x.len() as i32, x, 1)};
    if norm > f64::MIN_POSITIVE {
        scale(norm.recip(), x);
    }
}

fn inner_prod(x: &[f64], y: &[f64]) -> f64
{
    assert_eq!(x.len(), y.len());

    unsafe { cblas::ddot(x.len() as i32, x, 1, y, 1) }
}

fn copy(x: &[f64], y: &mut[f64])
{
    assert_eq!(x.len(), y.len());

    unsafe { cblas::dcopy(x.len() as i32, x, 1, y, 1) }
}

fn scale(alpha: f64, x: &mut[f64])
{
    unsafe { cblas::dscal(x.len() as i32, alpha, x, 1) }
}

fn add(alpha: f64, x: &[f64], y: &mut[f64])
{
    assert_eq!(x.len(), y.len());

    unsafe { cblas::daxpy(x.len() as i32, alpha, x, 1, y, 1) }
}

trait Operator
{
    fn size(&self) -> (usize, usize);
    // y = alpha * Op * x + beta * y
    fn op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64]);
    // y = alpha * Op^T * x + beta * y
    fn trans_op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64]);

    fn sp_norm(&self) -> f64
    {
        let mut v = vec![1.; self.size().1];
        let mut t = vec![0.; self.size().0];
        let mut w = vec![0.; self.size().1];
        let mut lambda = 0.;

        loop {
            normalize(&mut v);
            self.op(1., &v, 0., &mut t);
            self.trans_op(1., &t, 0., &mut w);

            let lambda_n = inner_prod(&v, &w);
    
            if (lambda_n - lambda).abs() <= f64::EPSILON {
                return lambda_n.sqrt();
            }
    
            copy(&w, &mut v);
            lambda = lambda_n;
        }
    }
}

struct Matrix<'a>
{
    n_row: usize,
    n_col: usize,
    array: &'a[f64],
}

impl<'a> Matrix<'a>
{
    fn new((n_row, n_col): (usize, usize), array: &'a[f64]) -> Self
    {
        assert_eq!(n_row * n_col, array.len());

        Matrix {
            n_row, n_col, array
        }
    }
}

impl<'a> Operator for Matrix<'a>
{
    fn size(&self) -> (usize, usize)
    {
        (self.n_row, self.n_col)
    }

    fn op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        assert_eq!(x.len(), self.n_col);
        assert_eq!(y.len(), self.n_row);
        
        unsafe { cblas::dgemv(
            cblas::Layout::RowMajor, cblas::Transpose::None,
            self.n_row as i32, self.n_col as i32,
            alpha, self.array, self.n_col as i32,
            x, 1,
            beta, y, 1
        ) }
    }

    fn trans_op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        assert_eq!(x.len(), self.n_row);
        assert_eq!(y.len(), self.n_col);
        
        unsafe { cblas::dgemv(
            cblas::Layout::RowMajor, cblas::Transpose::Ordinary,
            self.n_row as i32, self.n_col as i32,
            alpha, self.array, self.n_col as i32,
            x, 1,
            beta, y, 1
        ) }
    }
}

struct SelfDualEmbed<O: Operator>
{
    n: usize,
    m: usize,
    c: O,
    a: O,
    b: O,
}

impl<O: Operator> SelfDualEmbed<O>
{
    fn new(c: O, a: O, b: O) -> Self
    {
        let (m, n) = a.size();
         
        assert_eq!(c.size(), (n, 1));
        assert_eq!(b.size(), (m, 1));

        SelfDualEmbed {
            n, m, c, a, b
        }
    }
}

impl<O: Operator> Operator for SelfDualEmbed<O>
{
    fn size(&self) -> (usize, usize)
    {
        let nm1 = self.n + self.m + 1;

        (nm1, nm1 * 2)
    }

    fn op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        let nm1 = self.n + self.m + 1;

        if x.len() == nm1 * 2 {
            let (u_x, u) = x.split_at(self.n);
            let (u_y, u) = u.split_at(self.m);
            let (u_tau, v) = u.split_at(1);
            let (v_r, v) = v.split_at(self.n);
            let (v_s, v) = v.split_at(self.m);
            let (v_kappa, _) = v.split_at(1);

            let (w_n, w) = y.split_at_mut(self.n);
            let (w_m, w) = w.split_at_mut(self.m);
            let (w_1, _) = w.split_at_mut(1);

            self.a.trans_op(alpha, u_y, beta, w_n);
            self.c.op(alpha, u_tau, 1., w_n);
            add(-alpha, v_r, w_n);

            self.a.op(-alpha, u_x, beta, w_m);
            self.b.op(alpha, u_tau, 1., w_m);
            add(-alpha, v_s, w_m);

            self.c.trans_op(-alpha, u_x, beta, w_1);
            self.b.trans_op(-alpha, u_y, 1., w_1);
            add(-alpha, v_kappa, w_1);
        }
        else {
            // TODO
            assert!(false);
        }
    }

    fn trans_op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        let nm1 = self.n + self.m + 1;

        if y.len() == nm1 * 2 {
            let (w_n, w) = x.split_at(self.n);
            let (w_m, w) = w.split_at(self.m);
            let (w_1, _) = w.split_at(1);

            let (u_x, u) = y.split_at_mut(self.n);
            let (u_y, u) = u.split_at_mut(self.m);
            let (u_tau, v) = u.split_at_mut(1);
            let (v_r, v) = v.split_at_mut(self.n);
            let (v_s, v) = v.split_at_mut(self.m);
            let (v_kappa, _) = v.split_at_mut(1);

            self.a.trans_op(-alpha, w_m, beta, u_x);
            self.c.op(-alpha, w_1, 1., u_x);

            self.a.op(alpha, w_n, beta, u_y);
            self.b.op(-alpha, w_1, 1., u_y);

            self.c.trans_op(alpha, w_n, beta, u_tau);
            self.b.trans_op(alpha, w_m, 1., u_tau);

            scale(beta, v_r);
            add(-alpha, w_n, v_r);

            scale(beta, v_s);
            add(-alpha, w_m, v_s);

            scale(beta, v_kappa);
            add(-alpha, w_1, v_kappa);
        }
        else {
            // TODO
            assert!(false);
        }
    }
}

pub struct Solver;

impl Solver
{
    pub fn new() -> Self
    {
        Solver
    }

    pub fn solve(&self)
    {
        let max_iter = Some(5000);

        let n = 1;
        let m = 2;

        let mat_c = Matrix::new((n, 1), &[
            1.,
        ]);
        let mat_a = Matrix::new((m, n), &[
            -1.,
            1.,
        ]);
        let mat_b = Matrix::new((m, 1), &[
            2.,
            5.,
        ]);

        let op_l = SelfDualEmbed::new(mat_c, mat_a, mat_b);
        assert_eq!(op_l.sp_norm(), -1.);
    }
}
