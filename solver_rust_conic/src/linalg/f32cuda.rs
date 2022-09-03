use core::ops::{Index, IndexMut};
use std::vec::Vec;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::Arc;
use super::{SliceRef, SliceMut, SliceLike, LinAlg, LinAlgEx};
use crate::utils::*;
use rustacuda::prelude::*;
use rustacuda::memory::DeviceBuffer;
use once_cell::sync::Lazy;
//

#[derive(Eq, PartialEq, Copy, Clone, Debug)]
enum CUDASliceMut
{
    Sync,
    Host,
    Dev,
}

struct F32CUDABuf(DeviceBuffer<f32>);
unsafe impl Send for F32CUDABuf {}
unsafe impl Sync for F32CUDABuf {}

pub struct F32CUDASlice
{
    idx: usize,
    buf: Arc<F32CUDABuf>,
    sta: usize,
    end: usize,
    mutator: Mutex<CUDASliceMut>,
    vec: Mutex<Vec<f32>>,
}

struct SliceManager
{
    cnt: usize,
    map: HashMap<usize, F32CUDASlice>,
}

impl SliceManager
{
    fn new() -> Self
    {
        std::println!("{:#?}", CUDA_CONTEXT.0); // TODO
        SliceManager { cnt: 0, map: HashMap::new() }
    }

    fn new_slice(&mut self, s: &[f32]) -> &'static mut F32CUDASlice
    {
        let idx = self.cnt;
        self.cnt += 1;

        let cs = F32CUDASlice {
            idx,
            buf: Arc::new(F32CUDABuf(DeviceBuffer::from_slice(s).unwrap())),
            sta: 0,
            end: s.len(),
            mutator: Mutex::new(CUDASliceMut::Dev),
            vec: Mutex::new(Vec::new()),
        };

        self.map.insert(idx, cs);
        let cs = self.map.get_mut(&idx).unwrap();

        unsafe {
            core::mem::transmute::<&'_ mut F32CUDASlice, &'static mut F32CUDASlice>(cs)
        }
    }

    fn split_slice(&mut self, cso: &F32CUDASlice, sta: usize, end: usize) -> &'static mut F32CUDASlice
    {
        let idx = self.cnt;
        self.cnt += 1;

        let cs = F32CUDASlice {
            idx,
            buf: cso.buf.clone(),
            sta: cso.sta + sta,
            end: cso.sta + end,
            mutator: Mutex::new(CUDASliceMut::Dev),
            vec: Mutex::new(Vec::new()),
        };
        assert!(cs.sta <= cs.end);
        assert!(cs.end <= cso.end);

        self.map.insert(idx, cs);
        let cs = self.map.get_mut(&idx).unwrap();

        unsafe {
            core::mem::transmute::<&'_ mut F32CUDASlice, &'static mut F32CUDASlice>(cs)
        }
    }

    fn remove(&mut self, cs: &F32CUDASlice)
    {
        let idx = cs.idx;
        self.map.remove(&idx).unwrap();
    }
}

static SLICE_MANAGER: Lazy<Mutex<SliceManager>> = Lazy::new(|| {
    Mutex::new(SliceManager::new())
});


impl SliceLike for F32CUDASlice
{
    type F = f32;

    fn new(s: &[f32]) -> SliceRef<'_, F32CUDASlice>
    {
        let mut mgr = SLICE_MANAGER.lock().unwrap();
        let cs = mgr.new_slice(s);
        
        std::println!("{} new", cs.idx);
        SliceRef {s: cs}
    }

    fn new_mut(s: &mut[f32]) -> SliceMut<'_, F32CUDASlice>
    {
        let mut mgr = SLICE_MANAGER.lock().unwrap();
        let cs = mgr.new_slice(s);

        std::println!("{} new_mut", cs.idx);
        SliceMut {s: cs}
    }
    
    fn split_at(&self, mid: usize) -> (SliceRef<'_, F32CUDASlice>, SliceRef<'_, F32CUDASlice>)
    {
        let mut mgr = SLICE_MANAGER.lock().unwrap();
        let cs0 = mgr.split_slice(self, 0, mid);
        let cs1 = mgr.split_slice(self, mid, self.len());

        std::println!("{} split from {}", cs0.idx, self.idx);
        std::println!("{} split from {}", cs1.idx, self.idx);
        (SliceRef {s: cs0}, SliceRef {s: cs1})
    }

    fn split_at_mut(&mut self, mid: usize) -> (SliceMut<'_, F32CUDASlice>, SliceMut<'_, F32CUDASlice>)
    {
        let mut mgr = SLICE_MANAGER.lock().unwrap();
        let cs0 = mgr.split_slice(self, 0, mid);
        let cs1 = mgr.split_slice(self, mid, self.len());

        std::println!("{} split_mut from {}", cs0.idx, self.idx);
        std::println!("{} split_mut from {}", cs1.idx, self.idx);
        (SliceMut {s: cs0}, SliceMut {s: cs1})
    }

    fn drop(&self)
    {
        std::println!("{} drop", self.idx);
        let mut mgr = SLICE_MANAGER.lock().unwrap();
        mgr.remove(self);
    }

    fn len(&self) -> usize
    {
        std::println!("  {} len {}", self.idx, self.end - self.sta);
        self.end - self.sta
    }

    fn get(&self) -> &[f32]
    {
        std::println!("  {} get", self.idx);
        let mut mutator = self.mutator.lock().unwrap();
        std::println!("  {:?} mutator", *mutator);
        match *mutator {
            CUDASliceMut::Sync | CUDASliceMut::Host => {},
            CUDASliceMut::Dev => {
                let ds = &self.buf.0[self.sta..self.end];
                let mut vec = self.vec.lock().unwrap();

                vec.resize(self.len(), 0.);
                ds.copy_to(&mut *vec).unwrap();

                *mutator = CUDASliceMut::Sync;
            },
        }

        let vec = self.vec.lock().unwrap();
        let vec_ref: &[f32] = vec.as_ref();
        std::println!("  {}", vec_ref.len());

        unsafe {
            core::mem::transmute::<&'_ [f32], &'_ [f32]>(vec_ref)
        }
    }

    fn get_mut(&mut self) -> &mut[f32]
    {
        std::println!("  {} get_mut", self.idx);
        let mut mutator = self.mutator.lock().unwrap();
        std::println!("  {:?} mutator", *mutator);
        match *mutator {
            CUDASliceMut::Sync | CUDASliceMut::Host => {},
            CUDASliceMut::Dev => {
                let ds = &self.buf.0[self.sta..self.end];
                let mut vec = self.vec.lock().unwrap();

                vec.resize(self.len(), 0.);
                ds.copy_to(&mut *vec).unwrap();

                *mutator = CUDASliceMut::Host;
            },
        }

        let mut vec = self.vec.lock().unwrap();
        let vec_mut: &mut[f32] = vec.as_mut();
        std::println!("  {}", vec_mut.len());

        unsafe {
            core::mem::transmute::<&'_ mut[f32], &'_ mut[f32]>(vec_mut)
        }
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


#[derive(Debug)]
struct CudaContext(Context);

unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

static CUDA_CONTEXT: Lazy<CudaContext> = Lazy::new(|| {
    // Initialize the CUDA API
    rustacuda::init(CudaFlags::empty()).unwrap();

    // Get the first device
    let device = Device::get_device(0).unwrap();

    // Create a context associated to this device
    let context = Context::create_and_push(
        ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
        device
    ).unwrap();

    CudaContext(context)
});


struct CublasContext(cublas::Context);

unsafe impl Send for CublasContext {}
unsafe impl Sync for CublasContext {}


static CUBLAS_CONTEXT: Lazy<CublasContext> = Lazy::new(|| {
    let cublas_ctx = cublas::Context::new().unwrap();

    CublasContext(cublas_ctx)
});


impl LinAlg for F32CUDA
{
    type F = f32;
    type Sl = F32CUDASlice;

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

        let mut x_dev = DeviceBuffer::from_slice(x).unwrap();

        cublas::API::scal(&CUBLAS_CONTEXT.0,
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

impl LinAlgEx for F32CUDA
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
