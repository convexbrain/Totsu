use core::fmt::Debug;
use core::ops::{Index, IndexMut};
use core::cell::{Cell, RefCell};
use std::rc::Rc;
use std::vec::Vec;
use std::boxed::Box;
use super::{SliceRef, SliceMut, SliceDrop, SliceLike, LinMem, LinAlg, LinAlgEx};
use crate::utils::*;
use rustacuda::prelude::*;
use rustacuda::memory::DeviceBuffer;
//

#[derive(Eq, PartialEq, Copy, Clone)]
enum CUDASliceMut
{
    Sync,
    Host,
    Dev,
}

pub struct F32CUDASlice
{
    buf: Rc<RefCell<DeviceBuffer<f32>>>,
    sta: usize,
    end: usize,
    mutator: Cell<CUDASliceMut>,
    vec: Vec<f32>,
}

// TODO
static mut SLICE_LIST: Vec<Box<F32CUDASlice>> = Vec::new();


impl SliceDrop for F32CUDASlice {}

impl SliceLike<f32> for F32CUDASlice
{
    fn len(&self) -> usize
    {
        self.end - self.sta
    }

    fn split_at(&self, mid: usize) -> (&Self, &Self)
    {
        let s0 = F32CUDASlice {
            buf: self.buf.clone(),
            sta: self.sta,
            end: self.sta + mid,
            mutator: self.mutator.clone(),
            vec: Vec::new(),
        };
        let s1 = F32CUDASlice {
            buf: self.buf.clone(),
            sta: self.sta + mid,
            end: self.end,
            mutator: self.mutator.clone(),
            vec: Vec::new(),
        };

        // TODO
        let (ss0, ss1) = unsafe {
            SLICE_LIST.push(Box::new(s0));
            SLICE_LIST.push(Box::new(s1));

            let l = SLICE_LIST.len();

            (SLICE_LIST[l - 2].as_ref(), SLICE_LIST[l - 1].as_ref())
        };

        (ss0, ss1)
    }

    fn split_at_mut(&mut self, mid: usize) -> (&mut Self, &mut Self)
    {
        let s0 = F32CUDASlice {
            buf: self.buf.clone(),
            sta: self.sta,
            end: self.sta + mid,
            mutator: self.mutator.clone(),
            vec: Vec::new(),
        };
        let s1 = F32CUDASlice {
            buf: self.buf.clone(),
            sta: self.sta + mid,
            end: self.end,
            mutator: self.mutator.clone(),
            vec: Vec::new(),
        };

        // TODO
        let (ss0, ss1) = unsafe {
            SLICE_LIST.push(Box::new(s0));
            SLICE_LIST.push(Box::new(s1));

            let l = SLICE_LIST.len();

            (SLICE_LIST[l - 2].as_mut(), SLICE_LIST[l - 1].as_mut())
        };

        (ss0, ss1)
    }

    fn get(&self) -> &[f32]
    {
        // TODO: sync
        &self.vec
    }

    fn get_mut(&mut self) -> &mut[f32]
    {
        // TODO: sync
        &mut self.vec
    }
}

/* TODO
    fn sync_mut(&mut self, s: &mut[f32])
    {
        match self.mutator.get() {
            CUDAMemMut::Sync | CUDAMemMut::Host => {},
            CUDAMemMut::Dev => {
                let ds = &self.buf.borrow()[self.sta..self.end];
                ds.copy_to(s).unwrap();
            },
        }
        self.mutator.set(CUDAMemMut::Host);
    }

    fn sync_ref(&self, s: &[f32])
    {
        match self.mutator.get() {
            CUDAMemMut::Sync | CUDAMemMut::Host => {},
            CUDAMemMut::Dev => {
                let ds = &self.buf.borrow()[self.sta..self.end];

                let us = s as *const[f32] as *mut[f32];
                unsafe {
                    // TODO: undefeined behaviour
                    
                    let us = us.as_mut().unwrap();
                    ds.copy_to(us).unwrap();
                }

                self.mutator.set(CUDAMemMut::Sync);
            },
        }
    }

    fn sync_dev_mut(&mut self, s: &[f32])
    {
        match self.mutator.get() {
            CUDAMemMut::Sync | CUDAMemMut::Dev => {},
            CUDAMemMut::Host => {
                let ds = &mut self.buf.borrow_mut()[self.sta..self.end];
                ds.copy_from(s).unwrap();
            },
        }
        self.mutator.set(CUDAMemMut::Dev);
    }

    fn sync_dev_ref(&self, s: &[f32])
    {
        match self.mutator.get() {
            CUDAMemMut::Sync | CUDAMemMut::Dev => {},
            CUDAMemMut::Host => {
                let ds = &mut self.buf.borrow_mut()[self.sta..self.end];
                ds.copy_from(s).unwrap();
                self.mutator.set(CUDAMemMut::Sync);
            },
        }
    }
*/

/// TODO
pub struct F32CUDA;

// TODO
static mut CUDA_CONTEXT: Option<Context> = None;
static mut CUBLAS_CONTEXT: Option<cublas::Context> = None;

impl F32CUDA
{
    // TODO
    fn get_context() -> (&'static Context, &'static cublas::Context)
    {
        unsafe {
            if CUDA_CONTEXT.is_none() {
                std::println!("CUDA Init");

                // Initialize the CUDA API
                rustacuda::init(CudaFlags::empty()).unwrap();

                // Get the first device
                let device = Device::get_device(0).unwrap();

                // Create a context associated to this device
                let context = Context::create_and_push(
                    ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
                    device
                ).unwrap();

                CUDA_CONTEXT = Some(context);

                let cublas_ctx = cublas::Context::new().unwrap();
                CUBLAS_CONTEXT = Some(cublas_ctx);

            }
            (CUDA_CONTEXT.as_ref().unwrap(), CUBLAS_CONTEXT.as_ref().unwrap())
        }
    }
}


impl LinMem<f32> for F32CUDA
{
    type Slice = F32CUDASlice;

    fn slice_ref(s: &[f32]) -> SliceRef<'_, F32CUDASlice>
    {
        let s = F32CUDASlice {
            buf: Rc::new(RefCell::new(DeviceBuffer::from_slice(s).unwrap())),
            sta: 0,
            end: s.len(),
            mutator: Cell::new(CUDASliceMut::Sync),
            vec: Vec::new(),
        };

        // TODO
        let ss = unsafe {
            SLICE_LIST.push(Box::new(s));

            let l = SLICE_LIST.len();

            SLICE_LIST[l - 1].as_ref()
        };

        SliceRef {s: ss}
    }

    fn slice_mut(s: &mut[f32]) -> SliceMut<'_, F32CUDASlice>
    {
        let s = F32CUDASlice {
            buf: Rc::new(RefCell::new(DeviceBuffer::from_slice(s).unwrap())),
            sta: 0,
            end: s.len(),
            mutator: Cell::new(CUDASliceMut::Sync),
            vec: Vec::new(),
        };

        // TODO
        let ss = unsafe {
            SLICE_LIST.push(Box::new(s));

            let l = SLICE_LIST.len();

            SLICE_LIST[l - 1].as_mut()
        };

        SliceMut {s: ss}
    }
}

impl LinAlg<f32> for F32CUDA
{
    fn norm(x: &F32CUDASlice) -> f32
    {
        let x = x.get();

        let mut sum = 0.;
        for u in x {
            sum = sum + *u * *u;
        }
        sum.sqrt()
    }
    
    fn copy(x: &F32CUDASlice, y: &mut F32CUDASlice)
    {
        let x = x.get();
        let y = y.get_mut();

        assert_eq!(x.len(), y.len());
    
        for (u, v) in x.iter().zip(y) {
            *v = *u;
        }
    }

    fn scale(alpha: f32, x: &mut F32CUDASlice)
    {
        let x = x.get_mut();

        // TODO
        let (_, cublas_ctx) = Self::get_context();

        let mut x_dev = DeviceBuffer::from_slice(x).unwrap();

        cublas::API::scal(cublas_ctx,
            (&alpha as *const f32) as *mut f32,
            x_dev.as_mut_ptr(),
            x.len() as i32,
            None
        ).unwrap();

        x_dev.copy_to(x).unwrap();
    }
    
    fn add(alpha: f32, x: &F32CUDASlice, y: &mut F32CUDASlice)
    {
        let x = x.get();
        let y = y.get_mut();

        assert_eq!(x.len(), y.len());
    
        for (u, v) in x.iter().zip(y) {
            *v = *v + alpha * *u;
        }
    }

    fn adds(s: f32, y: &mut F32CUDASlice)
    {
        let y = y.get_mut();

        for v in y {
            *v = *v + s;
        }
    }
    
    fn abssum(x: &F32CUDASlice, incx: usize) -> f32
    {
        let x = x.get();

        if incx == 0 {
            0.
        }
        else {
            let mut sum = 0.;
            for u in x.chunks(incx) {
                sum = sum + u[0].abs();
            }
            sum
        }
    }

    fn transform_di(alpha: f32, mat: &F32CUDASlice, x: &F32CUDASlice, beta: f32, y: &mut F32CUDASlice)
    {
        let mat = mat.get();
        let x = x.get();
        let y = y.get_mut();

        assert_eq!(mat.len(), x.len());
        assert_eq!(mat.len(), y.len());

        for (i, v) in y.iter_mut().enumerate() {
            *v = alpha * mat[i] * x[i] + beta * *v;
        }
    }
}

//

struct MatIdx<'a>
{
    n_row: usize,
    n_col: usize,
    mat: &'a[f32],
    transpose: bool,
}

impl<'a> MatIdx<'a>
{
    fn idx(&self, (r, c): (usize, usize)) -> usize
    {
        let (r, c) = if !self.transpose {(r, c)} else {(c, r)};
        
        assert!(r < self.n_row);
        assert!(c < self.n_col);

        c * self.n_row + r
    }
}

impl<'a> Index<(usize, usize)> for MatIdx<'a>
{
    type Output = f32;

    fn index(&self, index: (usize, usize)) -> &Self::Output
    {
        &self.mat[self.idx(index)]
    }
}

//

struct MatIdxMut<'a>
{
    n_row: usize,
    n_col: usize,
    mat: &'a mut[f32],
    transpose: bool,
}

impl<'a> MatIdxMut<'a>
{
    fn col_vec(&self, c: usize) -> &[f32]
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
            *a = 0.;
        }
    }
}

impl<'a> Index<(usize, usize)> for MatIdxMut<'a>
{
    type Output = f32;

    fn index(&self, index: (usize, usize)) -> &Self::Output
    {
        let mat_idx = MatIdx {
            n_row: self.n_row,
            n_col: self.n_col,
            mat: self.mat,
            transpose: self.transpose,
        };
        
        &self.mat[mat_idx.idx(index)]
    }
}

impl<'a> IndexMut<(usize, usize)> for MatIdxMut<'a>
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output
    {
        let mat_idx = MatIdx {
            n_row: self.n_row,
            n_col: self.n_col,
            mat: self.mat,
            transpose: self.transpose,
        };
        
        &mut self.mat[mat_idx.idx(index)]
    }
}

//

struct SpMatIdx<'a>
{
    n: usize,
    mat: &'a[f32],
}

impl<'a> SpMatIdx<'a>
{
    fn idx(&self, (r, c): (usize, usize)) -> usize
    {
        assert!(r < self.n);
        assert!(c < self.n);

        let (r, c) = if r < c {(r, c)} else {(c, r)};

        c * (c + 1) / 2 + r
    }
}

impl<'a> Index<(usize, usize)> for SpMatIdx<'a>
{
    type Output = f32;

    fn index(&self, index: (usize, usize)) -> &Self::Output
    {
        &self.mat[self.idx(index)]
    }
}

//

struct SpMatIdxMut<'a>
{
    n: usize,
    mat: &'a mut[f32],
}

impl<'a> SpMatIdxMut<'a>
{
    fn clear(&mut self)
    {
        for a in self.mat.iter_mut() {
            *a = 0.;
        }
    }

    fn rank1op(&mut self, alpha: f32, x: &[f32])
    {
        assert_eq!(x.len(), self.n);

        for c in 0.. self.n {
            for r in 0..= c {
                self[(r, c)] = alpha * x[r] * x[c] + self[(r, c)];
            }
        }
    }
}

impl<'a> Index<(usize, usize)> for SpMatIdxMut<'a>
{
    type Output = f32;

    fn index(&self, index: (usize, usize)) -> &Self::Output
    {
        let sp_mat_idx = SpMatIdx {
            n: self.n,
            mat: self.mat,
        };
        
        &self.mat[sp_mat_idx.idx(index)]
    }
}

impl<'a> IndexMut<(usize, usize)> for SpMatIdxMut<'a>
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output
    {
        let sp_mat_idx = SpMatIdx {
            n: self.n,
            mat: self.mat,
        };
        
        &mut self.mat[sp_mat_idx.idx(index)]
    }
}

//

fn jacobi_eig(spmat_x: &mut SpMatIdxMut, mat_z: &mut MatIdxMut, eps: f32)
{
    let n = spmat_x.n;
    let tol = eps * eps;
    let f0 = 0.;
    let f1 = 1.;
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

fn eig_func<E>(spmat_x: &mut SpMatIdxMut, eps_zero: f32, work: &mut[f32], func: E)
where E: Fn(f32)->Option<f32>
{
    let f1 = 1.;

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

impl LinAlgEx<f32> for F32CUDA
{
    // y = a*mat*x + b*y
    fn transform_ge(transpose: bool, n_row: usize, n_col: usize, alpha: f32, mat: &F32CUDASlice, x: &F32CUDASlice, beta: f32, y: &mut F32CUDASlice)
    {
        let mat = mat.get();
        let x = x.get();
        let y = y.get_mut();

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
            let mut mat_x = 0.;
            for c in 0.. x.len() {
                mat_x = mat_x + mat[(r, c)] * x[c];
            }
            y[r] = alpha * mat_x + beta * y[r];
        }
    }

    // y = a*mat*x + b*y
    fn transform_sp(n: usize, alpha: f32, mat: &F32CUDASlice, x: &F32CUDASlice, beta: f32, y: &mut F32CUDASlice)
    {
        let mat = mat.get();
        let x = x.get();
        let y = y.get_mut();

        assert_eq!(mat.len(), n * (n + 1) / 2);

        assert_eq!(x.len(), n);
        assert_eq!(y.len(), n);

        let mat = SpMatIdx {
            n, mat,
        };

        for r in 0.. y.len() {
            let mut mat_x = 0.;
            for c in 0.. x.len() {
                mat_x = mat_x + mat[(r, c)] * x[c];
            }
            y[r] = alpha * mat_x + beta * y[r];
        }
    }

    fn proj_psd_worklen(sn: usize) -> usize
    {
        let n = ((((8 * sn + 1) as f32).sqrt() as usize) - 1) / 2;
        assert_eq!(n * (n + 1) / 2, sn);

        eig_func_worklen(n)
    }

    fn proj_psd(x: &mut F32CUDASlice, eps_zero: f32, work: &mut F32CUDASlice)
    {
        let x = x.get_mut();
        let work = work.get_mut();

        let f0 = 0.;
        let f1 = 1.;
        let f2: f32 = f1 + f1;
        let fsqrt2 = f2.sqrt();

        let sn = x.len();
        let n = ((((8 * sn + 1) as f32).sqrt() as usize) - 1) / 2;

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

    fn sqrt_spmat(mat: &mut F32CUDASlice, eps_zero: f32, work: &mut F32CUDASlice)
    {
        let mat = mat.get_mut();
        let work = work.get_mut();

        let f0 = 0.;

        let sn = mat.len();
        let n = ((((8 * sn + 1) as f32).sqrt() as usize) - 1) / 2;

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
