
trait Operator
{
    fn size(&self) -> (usize, usize);
    // y = alpha * Op * x + beta * y
    fn op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64]);
    // y = alpha * Op^T * x + beta * y
    fn trans_op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64]);
}

fn norm(x: &[f64]) -> f64
{
    unsafe { cblas::dnrm2(x.len() as i32, x, 1) }
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

// spectral norm
fn sp_norm<O: Operator>(op: &O) -> f64
{
    let mut v = vec![1.; op.size().1];
    let mut t = vec![0.; op.size().0];
    let mut w = vec![0.; op.size().1];
    let mut lambda = 0.;

    loop {
        let n = norm(&v);
        if n > f64::MIN_POSITIVE {
            scale(n.recip(), &mut v);
        }

        op.op(1., &v, 0., &mut t);
        op.trans_op(1., &t, 0., &mut w);

        let lambda_n = inner_prod(&v, &w);

        if (lambda_n - lambda).abs() <= f64::EPSILON {
            return lambda_n.sqrt();
        }

        copy(&w, &mut v);
        lambda = lambda_n;
    }
}

// Frobenius norm
fn fr_norm<O: Operator>(op: &O) -> f64
{
    let mut v = vec![0.; op.size().1];
    let mut t = vec![0.; op.size().0];
    let mut sq_norm = 0.;

    for row in 0.. v.len() {
        v[row] = 1.;
        op.op(1., &v, 0., &mut t);
        let n = norm(&t);
        sq_norm += n * n;
        v[row] = 0.;
    }

    sq_norm.sqrt()
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

struct SelfDualEmbed<OC: Operator, OA: Operator, OB: Operator>
{
    n: usize,
    m: usize,
    c: OC,
    a: OA,
    b: OB,
}

impl<OC, OA, OB> SelfDualEmbed<OC, OA, OB>
where OC: Operator, OA: Operator, OB: Operator
{
    fn new(c: OC, a: OA, b: OB) -> Self
    {
        let (m, n) = a.size();
         
        assert_eq!(c.size(), (n, 1));
        assert_eq!(b.size(), (m, 1));

        SelfDualEmbed {
            n, m, c, a, b
        }
    }

    fn c(&self) -> &OC
    {
        &self.c
    }

    fn a(&self) -> &OA
    {
        &self.a
    }

    fn b(&self) -> &OB
    {
        &self.b
    }
}

impl<OC, OA, OB> Operator for SelfDualEmbed<OC, OA, OB>
where OC: Operator, OA: Operator, OB: Operator
{
    fn size(&self) -> (usize, usize)
    {
        let nm1 = self.n + self.m + 1;

        (nm1, nm1 * 2)
    }

    fn op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        let n = self.n;
        let m = self.m;
        
        assert_eq!(y.len(), n + m + 1);

        let (u_x, u) = x.split_at(n);
        let (u_y, u) = u.split_at(m);
        let (u_tau, v) = u.split_at(1);
        let (v_r, v) = if v.len() == n + m + 1 {
            v.split_at(n)
        }
        else if v.len() == m + 1 {
            v.split_at(0)
        }
        else {
            assert!(false);
            v.split_at(0)
        };
        let (v_s, v) = v.split_at(m);
        let (v_kappa, _) = v.split_at(1);

        let (w_n, w) = y.split_at_mut(n);
        let (w_m, w) = w.split_at_mut(m);
        let (w_1, _) = w.split_at_mut(1);

        self.a.trans_op(alpha, u_y, beta, w_n);
        self.c.op(alpha, u_tau, 1., w_n);
        if v_r.len() == w_n.len() {
            add(-alpha, v_r, w_n);
        }

        self.a.op(-alpha, u_x, beta, w_m);
        self.b.op(alpha, u_tau, 1., w_m);
        add(-alpha, v_s, w_m);

        self.c.trans_op(-alpha, u_x, beta, w_1);
        self.b.trans_op(-alpha, u_y, 1., w_1);
        add(-alpha, v_kappa, w_1);
    }

    fn trans_op(&self, alpha: f64, x: &[f64], beta: f64, y: &mut[f64])
    {
        let n = self.n;
        let m = self.m;
        
        assert_eq!(x.len(), n + m + 1);

        let (w_n, w) = x.split_at(n);
        let (w_m, w) = w.split_at(m);
        let (w_1, _) = w.split_at(1);

        let (u_x, u) = y.split_at_mut(n);
        let (u_y, u) = u.split_at_mut(m);
        let (u_tau, v) = u.split_at_mut(1);
        let (v_r, v) = if v.len() == n + m + 1 {
            v.split_at_mut(n)
        }
        else if v.len() == m + 1 {
            v.split_at_mut(0)
        }
        else {
            assert!(false);
            v.split_at_mut(0)
        };
        let (v_s, v) = v.split_at_mut(m);
        let (v_kappa, _) = v.split_at_mut(1);

        self.a.trans_op(-alpha, w_m, beta, u_x);
        self.c.op(-alpha, w_1, 1., u_x);

        self.a.op(alpha, w_n, beta, u_y);
        self.b.op(-alpha, w_1, 1., u_y);

        self.c.trans_op(alpha, w_n, beta, u_tau);
        self.b.trans_op(alpha, w_m, 1., u_tau);

        if v_r.len() == w_n.len() {
            scale(beta, v_r);
            add(-alpha, w_n, v_r);
        }

        scale(beta, v_s);
        add(-alpha, w_m, v_s);

        scale(beta, v_kappa);
        add(-alpha, w_1, v_kappa);
    }
}

fn proj_pos(x: &mut[f64])
{
    for e in x {
        *e = e.max(0.);
    }
}

fn proj_o(x: &mut[f64])
{
    for e in x {
        *e = 0.;
    }
}

fn proj_r(x: &mut[f64])
{
    //
}

fn mat_to_vec(m: &[f64], v: &mut[f64])
{
    let l = v.len();
    let n = (m.len() as f64).sqrt() as usize;
    assert_eq!(m.len(), n * n);
    assert_eq!(n * (n + 1) / 2, l);

    let mut ref_m = m;
    let mut ref_v = v;

    for c in 0.. n {
        // upper triangular elements of symmetric matrix vectorized in row-wise
        let (r, spl_m) = ref_m.split_at(n);
        ref_m = spl_m;
        let (_, rc) = r.split_at(c);

        let (vc, spl_v) = ref_v.split_at_mut(n - c);
        ref_v = spl_v;
        copy(rc, vc);

        let (_, vct) = vc.split_at_mut(1);
        scale(2_f64.sqrt(), vct);
    }

    assert!(ref_m.is_empty());
    assert!(ref_v.is_empty());
}

fn vec_to_mat(v: &[f64], m: &mut[f64])
{
    let l = v.len();
    let n = (m.len() as f64).sqrt() as usize;
    assert_eq!(m.len(), n * n);
    assert_eq!(n * (n + 1) / 2, l);

    let mut ref_m = m;
    let mut ref_v = v;

    for c in 0.. n {
        // upper triangular elements of symmetric matrix vectorized in row-wise
        let (r, spl_m) = ref_m.split_at_mut(n);
        ref_m = spl_m;
        let (_, rc) = r.split_at_mut(c);

        let (vc, spl_v) = ref_v.split_at(n - c);
        ref_v = spl_v;
        copy(vc, rc);

        let (_, rct) = rc.split_at_mut(1);
        scale(0.5_f64.sqrt(), rct);
    }

    assert!(ref_m.is_empty());
    assert!(ref_v.is_empty());
}

fn proj_psd(x: &mut[f64])
{
    let l = x.len();
    let n = (((8 * l + 1) as f64).sqrt() as usize - 1) / 2;

    let mut a = vec![0.; n * n];

    vec_to_mat(x, &mut a);

    let mut m = 0;
    let mut w = vec![0.; n];
    let mut z = vec![0.; n * n];
    let mut null: Vec<i32> = vec![];

    let n = n as i32;
    unsafe {
        lapacke::dsyevr(
            lapacke::Layout::RowMajor, b'V', b'V',
            b'U', n, &mut a, n,
            0., f64::INFINITY, 0, 0, 0.,
            &mut m, &mut w,
            &mut z, n, &mut null);
    }

    for e in &mut a {
        *e = 0.;
    }
    for i in 0.. m as usize {
        let e = w[i];
        let (_, ref_z) = z.split_at(i);
        unsafe {
            cblas::dsyr(
                cblas::Layout::RowMajor, cblas::Part::Upper,
                n, e,
                ref_z, n,
                &mut a, n);
        }

    }

    mat_to_vec(&a, x);
}

fn proj_cone(x: &mut[f64])
{
    //proj_pos(x);
    proj_psd(x);
}

fn proj_cone_conj(x: &mut[f64])
{
    //proj_pos(x);
    proj_psd(x);
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
        let m = 3;

        let mat_c = Matrix::new((n, 1), &[
            1.,
        ]);
        let mat_a = Matrix::new((m, n), &[
            0.,
            -1. * 1.41421356,
            -3.,
        ]);
        let mat_b = Matrix::new((m, 1), &[
            1.,
            0. * 1.41421356,
            10.,
        ]);

        let op_l = SelfDualEmbed::new(mat_c, mat_a, mat_b);
        let op_l_norm = sp_norm(&op_l);
        assert!(op_l_norm >= f64::MIN_POSITIVE);

        let tau = op_l_norm.recip();
        let sigma = op_l_norm.recip();

        let eps_zero = 1e-12;
        let eps_pri = 1e-6;
        let eps_dual = 1e-6;
        let eps_gap = 1e-6;
        let eps_unbdd = 1e-6;
        let eps_infeas = 1e-6;

        let b_norm = fr_norm(op_l.b());
        let c_norm = fr_norm(op_l.c());

        let mut x = vec![0.; n + (m + 1) * 2];
        x[n + m] = 1.; // u_tau
        x[n + m + 1 + m] = 1.; // v_kappa
        let mut xx = x.clone();
        let mut y = vec![0.; n + m + 1];

        let mut p = vec![0.; m];
        let mut d = vec![0.; n];
        let mut one = vec![1.];

        let mut i = 0;
        loop {
            println!("----- {}", i);

            op_l.trans_op(-tau, &y, 1.0, &mut x);

            {
                let (_, u) = x.split_at_mut(n);
                let (u_y, u) = u.split_at_mut(m);
                let (u_tau, v) = u.split_at_mut(1);
                let (v_s, v) = v.split_at_mut(m);
                let (v_kappa, _) = v.split_at_mut(1);

                proj_cone_conj(u_y);
                proj_pos(u_tau);
                proj_cone(v_s);
                proj_pos(v_kappa);
            }

            add(-2., &x, &mut xx);
            op_l.op(-sigma, &xx, 1., &mut y);
            copy(&x, &mut xx);

            //println!("{:?}", x);
            //println!("{:?}", y);

            {
                let (u_x, u) = x.split_at(n);
                let (u_y, u) = u.split_at(m);
                let (u_tau, v) = u.split_at(1);
                let (v_s, _) = v.split_at(m);

                let u_tau = u_tau[0];

                if u_tau > eps_zero {
                    one[0] = 1.;

                    copy(v_s, &mut p);
                    op_l.b().op(-1., &one, u_tau.recip(), &mut p);
                    op_l.a().op(u_tau.recip(), u_x, 1., &mut p);
    
                    op_l.c().op(1., &one, 0., &mut d);
                    op_l.a().trans_op(u_tau.recip(), u_y, 1., &mut d);
    
                    op_l.c().trans_op(u_tau.recip(), u_x, 0., &mut one);
                    let g_x = one[0];
    
                    op_l.b().trans_op(u_tau.recip(), u_y, 0., &mut one);
                    let g_y = one[0];
    
                    let g = g_x + g_y;

                    let term_pri = norm(&p) <= eps_pri * (1. + b_norm);
                    let term_dual = norm(&d) <= eps_dual * (1. + c_norm);
                    let term_gap = g.abs() <= eps_gap * (1. + g_x.abs() + g_y.abs());
    
                    println!("{} {} {}", term_pri, term_dual, term_gap);

                    if term_pri && term_dual && term_gap {
                        println!("converged");
                        break;
                    }
                }
                else {
                    copy(v_s, &mut p);
                    op_l.a().op(1., u_x, 1., &mut p);

                    op_l.a().trans_op(1., u_y, 0., &mut d);

                    op_l.c().trans_op(-1., u_x, 0., &mut one);
                    let m_cx = one[0];

                    op_l.b().trans_op(-1., u_y, 0., &mut one);
                    let m_by = one[0];
        
                    let term_unbdd = (m_cx > eps_zero) && (
                        norm(&p) * c_norm <= eps_unbdd * m_cx
                    );
        
                    let term_infeas = (m_by > eps_zero) && (
                        norm(&d) * b_norm <= eps_infeas * m_by
                    );
        
                    println!("{} {}", term_unbdd, term_infeas);
        
                    if term_unbdd {
                        println!("unbounded");
                        break;
                    }
        
                    if term_infeas {
                        println!("infeasible");
                        break;
                    }
                }
            }

            i += 1;
            if let Some(max_i) = max_iter {
                if i >= max_i {
                    println!("timeover");
                    break;
                }
            }
        }

        let u_tau = x[n + m];
        if u_tau > eps_zero {
            scale(u_tau.recip(), &mut x);
            println!("{:?}", x);
        }
    }
}
