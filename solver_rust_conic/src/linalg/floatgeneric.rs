use num::Float;
use core::fmt::Debug;
use core::marker::PhantomData;
use core::ops::{Index, IndexMut};
use super::{LinAlg, LinAlgEx};
use crate::utils::*;

//

/// `num::Float`-generic [`LinAlgEx`] implementation
/// 
/// All numeric operations are written in pure Rust, but slow.
#[derive(Debug, Clone)]
pub struct FloatGeneric<F>
{
    ph_f: PhantomData<F>,
}

impl<F: Float> LinAlg<F> for FloatGeneric<F>
{
    fn norm(x: &[F]) -> F
    {
        let mut sum = F::zero();
        for u in x {
            sum = sum + *u * *u;
        }
        sum.sqrt()
    }
    
    fn copy(x: &[F], y: &mut[F])
    {
        assert_eq!(x.len(), y.len());
    
        for (u, v) in x.iter().zip(y) {
            *v = *u;
        }
    }
    
    fn scale(alpha: F, x: &mut[F])
    {
        for u in x {
            *u = alpha * *u;
        }
    }
    
    fn add(alpha: F, x: &[F], y: &mut[F])
    {
        assert_eq!(x.len(), y.len());
    
        for (u, v) in x.iter().zip(y) {
            *v = *v + alpha * *u;
        }
    }

    fn abssum(x: &[F]) -> F
    {
        let mut sum = F::zero();
        for u in x.iter() {
            sum = sum + u.abs();
        }
        sum
    }

    fn transform_di(alpha: F, mat: &[F], x: &[F], beta: F, y: &mut[F])
    {
        assert_eq!(mat.len(), x.len());
        assert_eq!(mat.len(), y.len());

        for (i, v) in y.iter_mut().enumerate() {
            *v = alpha * mat[i] * x[i] + beta * *v;
        }
    }
}

//

struct MatIdx<'a, F: Float>
{
    n_row: usize,
    n_col: usize,
    mat: &'a[F],
    transpose: bool,
}

impl<'a, F: Float> MatIdx<'a, F>
{
    fn index(&self, (r, c): (usize, usize)) -> usize
    {
        let (r, c) = if !self.transpose {(r, c)} else {(c, r)};
        
        assert!(r < self.n_row);
        assert!(c < self.n_col);

        c * self.n_row + r
    }
}

impl<'a, F: Float> Index<(usize, usize)> for MatIdx<'a, F>
{
    type Output = F;

    fn index(&self, index: (usize, usize)) -> &F
    {
        &self.mat[self.index(index)]
    }
}

//

struct MatIdxMut<'a, F: Float>
{
    n_row: usize,
    n_col: usize,
    mat: &'a mut[F],
    transpose: bool,
}

impl<'a, F: Float> MatIdxMut<'a, F>
{
    fn col_vec(&self, c: usize) -> &[F]
    {
        assert!(c < self.n_col);
        assert!(!self.transpose);

        let (_, v) = self.mat.split_at(c * self.n_row);
        let (v, _) = v.split_at(self.n_row);

        v
    }

    fn clear(&mut self)
    {
        for a in self.mat.iter_mut() {
            *a = F::zero();
        }
    }
}

impl<'a, F: Float> Index<(usize, usize)> for MatIdxMut<'a, F>
{
    type Output = F;

    fn index(&self, index: (usize, usize)) -> &F
    {
        let mat_idx = MatIdx {
            n_row: self.n_row,
            n_col: self.n_col,
            mat: self.mat,
            transpose: self.transpose,
        };
        
        &self.mat[mat_idx.index(index)]
    }
}

impl<'a, F: Float> IndexMut<(usize, usize)> for MatIdxMut<'a, F>
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut F
    {
        let mat_idx = MatIdx {
            n_row: self.n_row,
            n_col: self.n_col,
            mat: self.mat,
            transpose: self.transpose,
        };
        
        &mut self.mat[mat_idx.index(index)]
    }
}

//

struct SpMatIdx<'a, F: Float>
{
    n: usize,
    mat: &'a[F],
}

impl<'a, F: Float> SpMatIdx<'a, F>
{
    fn index(&self, (r, c): (usize, usize)) -> usize
    {
        assert!(r < self.n);
        assert!(c < self.n);

        let (r, c) = if r < c {(r, c)} else {(c, r)};

        c * (c + 1) / 2 + r
    }
}

impl<'a, F: Float> Index<(usize, usize)> for SpMatIdx<'a, F>
{
    type Output = F;

    fn index(&self, index: (usize, usize)) -> &F
    {
        &self.mat[self.index(index)]
    }
}

//

struct SpMatIdxMut<'a, F: Float>
{
    n: usize,
    mat: &'a mut[F],
}

impl<'a, F: Float> SpMatIdxMut<'a, F>
{
    fn clear(&mut self)
    {
        for a in self.mat.iter_mut() {
            *a = F::zero();
        }
    }

    fn rank1op(&mut self, alpha: F, x: &[F])
    {
        assert_eq!(x.len(), self.n);

        for c in 0.. self.n {
            for r in 0..= c {
                self[(r, c)] = alpha * x[r] * x[c] + self[(r, c)];
            }
        }
    }
}

impl<'a, F: Float> Index<(usize, usize)> for SpMatIdxMut<'a, F>
{
    type Output = F;

    fn index(&self, index: (usize, usize)) -> &F
    {
        let sp_mat_idx = SpMatIdx {
            n: self.n,
            mat: self.mat,
        };
        
        &self.mat[sp_mat_idx.index(index)]
    }
}

impl<'a, F: Float> IndexMut<(usize, usize)> for SpMatIdxMut<'a, F>
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut F
    {
        let sp_mat_idx = SpMatIdx {
            n: self.n,
            mat: self.mat,
        };
        
        &mut self.mat[sp_mat_idx.index(index)]
    }
}

//

fn jacobi_eig<F: Float>(spmat_x: &mut SpMatIdxMut<F>, mat_z: &mut MatIdxMut<F>, eps: F)
{
    let n = spmat_x.n;
    let tol = eps * eps;
    let f0 = F::zero();
    let f1 = F::one();
    let f2 = f1 + f1;

    let mut conv = false;

    while !conv {
        conv = true;

        for i in 0.. n {
            for j in i + 1.. n {
                let a = spmat_x[(i, i)];
                let b = spmat_x[(j, j)];
                let d = spmat_x[(i, j)];

                if (d * d > tol * a * b) && (d * d > tol) {
                    conv = false;

                    let zeta = (b - a) / (f2 * d);
                    let t = if zeta > f0 {
                        f1 / (zeta + (f1 + zeta * zeta).sqrt())
                    }
                    else {
                        -f1 / (-zeta + (f1 + zeta * zeta).sqrt())
                    };
                    let c = (f1 + t * t).sqrt().recip();
                    let s = c * t;

                    for k in 0.. n {
                        let xi = spmat_x[(k, i)];
                        let xj = spmat_x[(k, j)];
                        spmat_x[(k, i)] = c * xi - s * xj;
                        spmat_x[(k, j)] = s * xi + c * xj;

                        let zi = mat_z[(k, i)];
                        let zj = mat_z[(k, j)];
                        mat_z[(k, i)] = c * zi - s * zj;
                        mat_z[(k, j)] = s * zi + c * zj;
                    }

                    spmat_x[(i, i)] = c * c * a + s * s * b - f2 * c * s * d;
                    spmat_x[(j, j)] = s * s * a + c * c * b + f2 * c * s * d;
                    spmat_x[(i, j)] = f0;
                }
            }
        }
    }
}

fn eig_func<F: Float, E>(spmat_x: &mut SpMatIdxMut<F>, eps_zero: F, work: &mut[F], func: E)
where E: Fn(F)->Option<F>
{
    let f1 = F::one();

    let n = spmat_x.n;

    let (w, z) = work.split2(n, n * n).unwrap();

    let mut mat_z = MatIdxMut {
        n_row: n, n_col: n, mat: z, transpose: false,
    };

    mat_z.clear();
    for i in 0.. n {
        mat_z[(i, i)] = f1;
    }

    jacobi_eig(spmat_x, &mut mat_z, eps_zero);

    for i in 0.. n {
        w[i] = spmat_x[(i, i)];
    }

    spmat_x.clear();
    for i in 0.. n {
        if let Some(e) = func(w[i]) {
            let zcol = mat_z.col_vec(i);
            spmat_x.rank1op(e, zcol);
        }
    }
}

fn eig_func_worklen(n: usize) -> usize
{
    let len_w = n;
    let len_z = n * n;

    len_w + len_z
}

//

impl<F: Float> LinAlgEx<F> for FloatGeneric<F>
{
    // y = a*mat*x + b*y
    fn transform_ge(transpose: bool, n_row: usize, n_col: usize, alpha: F, mat: &[F], x: &[F], beta: F, y: &mut[F])
    {
        assert_eq!(mat.len(), n_row * n_col);
        if transpose {
            assert_eq!(x.len(), n_row);
            assert_eq!(y.len(), n_col);
        } else {
            assert_eq!(x.len(), n_col);
            assert_eq!(y.len(), n_row);
        };

        let mat = MatIdx {
            n_row, n_col, mat, transpose,
        };

        for r in 0.. y.len() {
            let mut mat_x = F::zero();
            for c in 0.. x.len() {
                mat_x = mat_x + mat[(r, c)] * x[c];
            }
            y[r] = alpha * mat_x + beta * y[r];
        }
    }

    // y = a*mat*x + b*y
    fn transform_sp(n: usize, alpha: F, mat: &[F], x: &[F], beta: F, y: &mut[F])
    {
        assert_eq!(mat.len(), n * (n + 1) / 2);

        assert_eq!(x.len(), n);
        assert_eq!(y.len(), n);

        let mat = SpMatIdx {
            n, mat,
        };

        for r in 0.. y.len() {
            let mut mat_x = F::zero();
            for c in 0.. x.len() {
                mat_x = mat_x + mat[(r, c)] * x[c];
            }
            y[r] = alpha * mat_x + beta * y[r];
        }
    }

    fn proj_psd_worklen(sn: usize) -> usize
    {
        let n = (F::from(8 * sn + 1).unwrap().sqrt().to_usize().unwrap() - 1) / 2;
        assert_eq!(n * (n + 1) / 2, sn);

        eig_func_worklen(n)
    }

    fn proj_psd(x: &mut[F], eps_zero: F, work: &mut[F])
    {
        let f0 = F::zero();
        let f1 = F::one();
        let f2 = f1 + f1;
        let fsqrt2 = f2.sqrt();

        let sn = x.len();
        let n = (F::from(8 * sn + 1).unwrap().sqrt().to_usize().unwrap() - 1) / 2;

        assert!(work.len() >= Self::proj_psd_worklen(sn));

        let mut spmat_x = SpMatIdxMut {
            n, mat: x,
        };

        // scale diagonals to match the resulted matrix norm with the vector norm multiplied by 2
        for i in 0.. n {
            spmat_x[(i, i)] = spmat_x[(i, i)] * fsqrt2;
        }

        eig_func(&mut spmat_x, eps_zero, work, |e| {
            if e > f0 {
                Some(e)
            }
            else {
                None
            }
        });

        // scale diagonals to match the resulted vector norm with the matrix norm multiplied by 0.5
        for i in 0.. n {
            spmat_x[(i, i)] = spmat_x[(i, i)] * fsqrt2.recip();
        }
    }

    fn sqrt_spmat_worklen(n: usize) -> usize
    {
        eig_func_worklen(n)
    }

    fn sqrt_spmat(mat: &mut[F], eps_zero: F, work: &mut[F])
    {
        let f0 = F::zero();

        let sn = mat.len();
        let n = (F::from(8 * sn + 1).unwrap().sqrt().to_usize().unwrap() - 1) / 2;

        assert!(work.len() >= Self::proj_psd_worklen(sn));

        let mut spmat_x = SpMatIdxMut {
            n, mat,
        };

        eig_func(&mut spmat_x, eps_zero, work, |e| {
            if e > f0 {
                Some(e.sqrt())
            }
            else {
                None
            }
        });
    }
}
