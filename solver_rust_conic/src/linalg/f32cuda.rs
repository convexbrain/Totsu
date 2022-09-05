use core::ops::{Index, IndexMut};
use std::vec::Vec;
use std::collections::HashMap;
use std::sync::{Mutex, Arc};
use std::boxed::Box;
use std::pin::Pin;
use super::{SliceRef, SliceMut, SliceLike, LinAlg, LinAlgEx};
use crate::utils::SplitN;
use rustacuda::prelude::*;
use rustacuda::memory::{DeviceBuffer, DeviceSlice};
use cublas_sys::*;
use once_cell::sync::Lazy;

//

/// TODO: doc
pub struct CudaManager
{
    cuda_ctx: Context,
    cublas_handle: cublasHandle_t,
}

unsafe impl Send for CudaManager {}
unsafe impl Sync for CudaManager {}

impl CudaManager
{
    fn buf_from_slice(&self, s: &[f32]) -> DeviceBuffer<f32>
    {
        DeviceBuffer::from_slice(s).unwrap()
    }

    /// TODO: doc
    pub fn context(&self) -> &Context
    {
        &self.cuda_ctx
    }

    /// TODO: doc
    pub fn cublas_handle(&self) -> &cublasHandle_t
    {
        &self.cublas_handle
    }
}

/// TODO: doc
pub static CUDA_MANAGER: Lazy<CudaManager> = Lazy::new(|| {
    // Initialize the CUDA API
    let r = rustacuda::init(CudaFlags::empty());
    if r.is_err() {
        log::error!("CUDA driver initialization failed");
    }
    r.unwrap();

    // API version
    log::info!(
        "CUDA driver API version: {}.{}",
        rustacuda::CudaApiVersion::get().unwrap().major(),
        rustacuda::CudaApiVersion::get().unwrap().minor(),
    );

    // Get the first device
    // TODO: num_devices
    let device = Device::get_device(0);
    if device.is_err() {
        log::error!("CUDA device not found");
    }
    let device = device.unwrap();

    // Device name
    log::info!("CUDA device name: {}", device.name().unwrap());

    // Create a context associated to this device
    let cuda_ctx = Context::create_and_push(
        ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
        device
    );
    if cuda_ctx.is_err() {
        log::error!("CUDA context failed to create");
    }
    let cuda_ctx = cuda_ctx.unwrap();

    // cuBLAS context
    let mut cublas_handle: cublasHandle_t = core::ptr::null_mut();
    unsafe {
        let st = cublasCreate_v2(&mut cublas_handle);
        if st != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            log::error!("cuBLAS handle failed to create");
        }
        assert_eq!(st, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    }

    log::debug!("CUDA_MANAGER created");
    CudaManager {
        cuda_ctx,
        cublas_handle,
    }
});



#[derive(Eq, PartialEq, Copy, Clone)]
enum CUDASliceMut
{
    Sync,
    Host,
    Dev,
}

struct F32CUDABuf(DeviceBuffer<f32>);
unsafe impl Send for F32CUDABuf {}
unsafe impl Sync for F32CUDABuf {}

/// TODO: doc
pub struct F32CUDASlice
{
    idx: usize,
    parent_idx: Option<usize>,
    dev_buf: Arc<Mutex<F32CUDABuf>>,
    host_buf: Arc<Mutex<Vec<f32>>>,
    sta: usize,
    end: usize,
    mutator: Mutex<CUDASliceMut>,
}

struct SliceManager
{
    cnt: usize,
    map: HashMap<usize, Pin<Box<F32CUDASlice>>>,
}

impl SliceManager
{
    fn new_slice<'a>(&mut self, s: &'a[f32]) -> &'a mut F32CUDASlice
    {
        let idx = self.cnt;
        self.cnt = idx + 1;

        let cs = F32CUDASlice {
            idx,
            parent_idx: None,
            dev_buf: Arc::new(Mutex::new(F32CUDABuf(CUDA_MANAGER.buf_from_slice(s)))),
            host_buf: Arc::new(Mutex::new(Vec::from(s))),
            sta: 0,
            end: s.len(),
            mutator: Mutex::new(CUDASliceMut::Sync),
        };

        let r = self.map.insert(idx, Box::pin(cs));
        assert!(r.is_none());
        let cs = self.map.get_mut(&idx).unwrap();

        unsafe {
            core::mem::transmute::<&mut F32CUDASlice, &'a mut F32CUDASlice>(cs)
        }
    }

    fn split_slice<'a>(&mut self, cso: &'a F32CUDASlice, sta: usize, end: usize) -> &'a mut F32CUDASlice
    {
        assert!(sta <= end, "{}<={}", sta, end);
        let idx = self.cnt;
        self.cnt = idx + 1;

        let cs = F32CUDASlice {
            idx,
            parent_idx: Some(cso.idx),
            dev_buf: cso.dev_buf.clone(),
            host_buf: cso.host_buf.clone(),
            sta: cso.sta + sta,
            end: cso.sta + end,
            mutator: Mutex::new(*cso.mutator.lock().unwrap()),
        };
        assert!(cs.sta <= cs.end);
        assert!(cs.end <= cso.end);

        let r = self.map.insert(idx, Box::pin(cs));
        assert!(r.is_none());
        let cs = self.map.get_mut(&idx).unwrap();

        unsafe {
            core::mem::transmute::<&mut F32CUDASlice, &'a mut F32CUDASlice>(cs)
        }
    }

    fn remove(&mut self, idx: usize)
    {
        // sync mutators of split tree
        //
        // parent    child(cs)
        // Sync   -> Sync       do nothing
        //        -> Host       change parent to Host
        //        -> Dev        change parent to Dev
        // Host   -> Sync       do nothing
        //        -> Host       do nothing
        //        -> Dev        copy from Dev to Host
        // Dev    -> Sync       do nothing
        //        -> Host       copy from Host to Dev
        //        -> Dev        do nothing
        let cs = &self.map[&idx];
        let cs_mut = *cs.mutator.lock().unwrap();
        let cs_par_idx = cs.parent_idx;
        let mut cs_par_mut = None;

        if let Some(par_idx) = cs_par_idx {
            let par = self.map.get_mut(&par_idx).unwrap();
            let mut par_mut = par.mutator.lock().unwrap();

            cs_par_mut = Some(*par_mut);

            if *par_mut == CUDASliceMut::Sync {
                *par_mut = cs_mut;
            }
        }

        let cs = self.map.get_mut(&idx).unwrap();
        if let Some(par_mut) = cs_par_mut {
            match par_mut {
                CUDASliceMut::Sync => {},
                CUDASliceMut::Host => {
                    if *cs.mutator.lock().unwrap() == CUDASliceMut::Dev {
                        let db = &cs.dev_buf.lock().unwrap().0[cs.sta..cs.end];
                        let mut hb = cs.host_buf.lock().unwrap();
                        let mut hb_mut = &mut hb[cs.sta..cs.end];
                        
                        db.copy_to(&mut hb_mut).unwrap();
                    }
                },
                CUDASliceMut::Dev => {
                    if *cs.mutator.lock().unwrap() == CUDASliceMut::Host {
                        let db = &mut cs.dev_buf.lock().unwrap().0[cs.sta..cs.end];
                        let mut hb = cs.host_buf.lock().unwrap();
                        let mut hb_mut = &mut hb[cs.sta..cs.end];
                        
                        db.copy_from(&mut hb_mut).unwrap();
                    }
                },
            }
        }
        self.map.remove(&idx).unwrap();
    }
}

impl Drop for SliceManager
{
    fn drop(&mut self) {
        assert_eq!(self.map.len(), 0);
    }
}

static SLICE_MANAGER: Lazy<Mutex<SliceManager>> = Lazy::new(|| {
    Mutex::new(SliceManager {
        cnt: 0,
        map: HashMap::new()
    })
});

impl SliceLike for F32CUDASlice
{
    type F = f32;

    fn new(s: &[f32]) -> SliceRef<'_, F32CUDASlice>
    {
        let mut mgr = SLICE_MANAGER.lock().unwrap();
        let cs = mgr.new_slice(s);
        
        //std::println!("{} new", cs.idx);
        SliceRef {s: cs}
    }

    fn new_mut(s: &mut[f32]) -> SliceMut<'_, F32CUDASlice>
    {
        let mut mgr = SLICE_MANAGER.lock().unwrap();
        let cs = mgr.new_slice(s);

        //std::println!("{} new_mut", cs.idx);
        SliceMut {s: cs}
    }
    
    fn split_at(&self, mid: usize) -> (SliceRef<'_, F32CUDASlice>, SliceRef<'_, F32CUDASlice>)
    {
        let mut mgr = SLICE_MANAGER.lock().unwrap();
        let cs0 = mgr.split_slice(self, 0, mid);
        let cs1 = mgr.split_slice(self, mid, self.len());

        //std::println!("{} split from {}", cs0.idx, self.idx);
        //std::println!("{} split from {}", cs1.idx, self.idx);
        (SliceRef {s: cs0}, SliceRef {s: cs1})
    }

    fn split_at_mut(&mut self, mid: usize) -> (SliceMut<'_, F32CUDASlice>, SliceMut<'_, F32CUDASlice>)
    {
        let mut mgr = SLICE_MANAGER.lock().unwrap();
        let cs0 = mgr.split_slice(self, 0, mid);
        let cs1 = mgr.split_slice(self, mid, self.len());

        //std::println!("{} split_mut from {}", cs0.idx, self.idx);
        //std::println!("{} split_mut from {}", cs1.idx, self.idx);
        (SliceMut {s: cs0}, SliceMut {s: cs1})
    }

    fn drop(&self)
    {
        //std::println!("{} drop", self.idx);
        let mut mgr = SLICE_MANAGER.lock().unwrap();
        mgr.remove(self.idx);
    }

    fn len(&self) -> usize
    {
        //std::println!("  {} len {}", self.idx, self.end - self.sta);
        self.end - self.sta
    }

    fn get(&self) -> &[f32]
    {
        //std::println!("  {} get", self.idx);
        let mut mutator = self.mutator.lock().unwrap();
        //std::println!("  {:?} mutator", *mutator);
        match *mutator {
            CUDASliceMut::Sync | CUDASliceMut::Host => {},
            CUDASliceMut::Dev => {
                let db = &self.dev_buf.lock().unwrap().0[self.sta..self.end];
                let mut hb = &mut self.host_buf.lock().unwrap()[self.sta..self.end];
                
                db.copy_to(&mut hb).unwrap();

                *mutator = CUDASliceMut::Sync;
            },
        }

        let hb_ref = &self.host_buf.lock().unwrap()[self.sta..self.end];

        unsafe {
            core::mem::transmute::<&[f32], &[f32]>(hb_ref)
        }
    }

    fn get_mut(&mut self) -> &mut[f32]
    {
        //std::println!("  {} get_mut", self.idx);
        let mut mutator = self.mutator.lock().unwrap();
        //std::println!("    {:?} mutator", *mutator);
        match *mutator {
            CUDASliceMut::Sync => {
                *mutator = CUDASliceMut::Host;
            },
            CUDASliceMut::Host => {},
            CUDASliceMut::Dev => {
                let db = &self.dev_buf.lock().unwrap().0[self.sta..self.end];
                let mut hb = &mut self.host_buf.lock().unwrap()[self.sta..self.end];

                db.copy_to(&mut hb).unwrap();

                *mutator = CUDASliceMut::Host;
            },
        }

        let hb_mut = &mut self.host_buf.lock().unwrap()[self.sta..self.end];

        unsafe {
            core::mem::transmute::<&mut[f32], &mut[f32]>(hb_mut)
        }
    }
}

impl F32CUDASlice
{
    /// TODO: doc
    pub fn get_dev(&self) -> &DeviceSlice<f32>
    {
        let mut mutator = self.mutator.lock().unwrap();
        match *mutator {
            CUDASliceMut::Sync | CUDASliceMut::Dev => {},
            CUDASliceMut::Host => {
                let db = &mut self.dev_buf.lock().unwrap().0[self.sta..self.end];
                let mut hb = &mut self.host_buf.lock().unwrap()[self.sta..self.end];
                
                db.copy_from(&mut hb).unwrap();

                *mutator = CUDASliceMut::Sync;
            },
        }

        let db_ref = &self.dev_buf.lock().unwrap().0[self.sta..self.end];

        unsafe {
            core::mem::transmute::<&DeviceSlice<f32>, &DeviceSlice<f32>>(db_ref)
        }
    }

    /// TODO: doc
    pub fn get_dev_mut(&mut self) -> &mut DeviceSlice<f32>
    {
        let mut mutator = self.mutator.lock().unwrap();
        match *mutator {
            CUDASliceMut::Sync => {
                *mutator = CUDASliceMut::Dev;
            },
            CUDASliceMut::Dev => {},
            CUDASliceMut::Host => {
                let db = &mut self.dev_buf.lock().unwrap().0[self.sta..self.end];
                let mut hb = &mut self.host_buf.lock().unwrap()[self.sta..self.end];
                
                db.copy_from(&mut hb).unwrap();

                *mutator = CUDASliceMut::Dev;
            },
        }

        let db_mut = &mut self.dev_buf.lock().unwrap().0[self.sta..self.end];

        unsafe {
            core::mem::transmute::<&mut DeviceSlice<f32>, &mut DeviceSlice<f32>>(db_mut)
        }
    }
}

/// TODO: doc
pub struct F32CUDA;

//--------------------------------------------------

impl LinAlg for F32CUDA
{
    type F = f32;
    type Sl = F32CUDASlice;

    fn norm(x: &F32CUDASlice) -> f32
    {
        let mut result = 0.;

        unsafe {
            let st = cublasSnrm2_v2(
                *CUDA_MANAGER.cublas_handle(),
                x.len() as i32,
                x.get_dev().as_ptr(), 1,
                &mut result
            );
            assert_eq!(st, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        }
        
        result
    }
    
    fn copy(x: &F32CUDASlice, y: &mut F32CUDASlice)
    {
        assert_eq!(x.len(), y.len());

        unsafe {
            let st = cublasScopy_v2(
                *CUDA_MANAGER.cublas_handle(),
                x.len() as i32,
                x.get_dev().as_ptr(), 1,
                y.get_dev_mut().as_mut_ptr(), 1
            );
            assert_eq!(st, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        }
    }

    fn scale(alpha: f32, x: &mut F32CUDASlice)
    {
        unsafe {
            let st = cublasSscal_v2(
                *CUDA_MANAGER.cublas_handle(),
                x.len() as i32,
                &alpha, x.get_dev_mut().as_mut_ptr(), 1
            );
            assert_eq!(st, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        }
    }
    
    fn add(alpha: f32, x: &F32CUDASlice, y: &mut F32CUDASlice)
    {
        assert_eq!(x.len(), y.len());

        unsafe {
            let st = cublasSaxpy_v2(
                *CUDA_MANAGER.cublas_handle(),
                x.len() as i32,
                &alpha, x.get_dev().as_ptr(), 1,
                y.get_dev_mut().as_mut_ptr(), 1
            );
            assert_eq!(st, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        }
    }

    fn adds(s: f32, y: &mut F32CUDASlice)
    {
        let one = CUDA_MANAGER.buf_from_slice(&[1.]);

        unsafe {
            let st = cublasSaxpy_v2(
                *CUDA_MANAGER.cublas_handle(),
                y.len() as i32,
                &s, one.as_ptr(), 0,
                y.get_dev_mut().as_mut_ptr(), 1
            );
            assert_eq!(st, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        }
    }
    
    fn abssum(x: &F32CUDASlice, incx: usize) -> f32
    {
        if incx == 0 {
            0.
        }
        else {
            let mut result = 0.;

            unsafe {
                let st = cublasSasum_v2(
                    *CUDA_MANAGER.cublas_handle(),
                    ((x.len() + (incx - 1)) / incx) as i32,
                    x.get_dev().as_ptr(), incx as i32,
                    &mut result
                );
                assert_eq!(st, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
            }

            result
        }
    }

    fn transform_di(alpha: f32, mat: &F32CUDASlice, x: &F32CUDASlice, beta: f32, y: &mut F32CUDASlice)
    {
        unsafe {
            let st = cublasSsbmv_v2(
                *CUDA_MANAGER.cublas_handle(),
                cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
                mat.len() as i32, 0,
                &alpha, mat.get_dev().as_ptr(), 1,
                x.get_dev().as_ptr(), 1,
                &beta, y.get_dev_mut().as_mut_ptr(), 1
            );
            assert_eq!(st, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
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
        assert_eq!(mat.len(), n_row * n_col);

        let trans = if transpose {
            assert_eq!(x.len(), n_row);
            assert_eq!(y.len(), n_col);

            cublasOperation_t::CUBLAS_OP_T
        } else {
            assert_eq!(x.len(), n_col);
            assert_eq!(y.len(), n_row);

            cublasOperation_t::CUBLAS_OP_N
        };

        unsafe {
            let st = cublasSgemv_v2(
                *CUDA_MANAGER.cublas_handle(),
                trans,
                n_row as i32, n_col as i32,
                &alpha, mat.get_dev().as_ptr(), n_row as i32,
                x.get_dev().as_ptr(), 1,
                &beta, y.get_dev_mut().as_mut_ptr(), 1
            );
            assert_eq!(st, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        }
    }

    // y = a*mat*x + b*y
    fn transform_sp(n: usize, alpha: f32, mat: &F32CUDASlice, x: &F32CUDASlice, beta: f32, y: &mut F32CUDASlice)
    {
        // TODO: cuda test
        /*
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
        */
        unsafe {
            let st = cublasSspmv_v2(
                *CUDA_MANAGER.cublas_handle(),
                cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
                n as i32,
                &alpha, mat.get_dev().as_ptr(),
                x.get_dev().as_ptr(), 1,
                &beta, y.get_dev_mut().as_mut_ptr(), 1
            );
            assert_eq!(st, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        }
    }

    fn proj_psd_worklen(sn: usize) -> usize
    {
        // TODO: cusolver
        let n = ((((8 * sn + 1) as f32).sqrt() as usize) - 1) / 2;
        assert_eq!(n * (n + 1) / 2, sn);

        eig_func_worklen(n)
    }

    fn proj_psd(x: &mut F32CUDASlice, eps_zero: f32, work: &mut F32CUDASlice)
    {
        // TODO: cusolver
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
        // TODO: cusolver
        eig_func_worklen(n)
    }

    fn sqrt_spmat(mat: &mut F32CUDASlice, eps_zero: f32, work: &mut F32CUDASlice)
    {
        // TODO: cusolver
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

//

pub mod cusolver_sys_partial
{
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]

    use super::cublasFillMode_t;

    #[derive(PartialEq, Debug, Clone, Copy)]
    #[repr(u32)]
    pub enum cusolverStatus_t {
        CUSOLVER_STATUS_SUCCESS=0,
        CUSOLVER_STATUS_NOT_INITIALIZED=1,
        CUSOLVER_STATUS_ALLOC_FAILED=2,
        CUSOLVER_STATUS_INVALID_VALUE=3,
        CUSOLVER_STATUS_ARCH_MISMATCH=4,
        CUSOLVER_STATUS_MAPPING_ERROR=5,
        CUSOLVER_STATUS_EXECUTION_FAILED=6,
        CUSOLVER_STATUS_INTERNAL_ERROR=7,
        CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED=8,
        CUSOLVER_STATUS_NOT_SUPPORTED = 9,
        CUSOLVER_STATUS_ZERO_PIVOT=10,
        CUSOLVER_STATUS_INVALID_LICENSE=11,
        CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED=12,
        CUSOLVER_STATUS_IRS_PARAMS_INVALID=13,
        CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC=14,
        CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE=15,
        CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER=16,
        CUSOLVER_STATUS_IRS_INTERNAL_ERROR=20,
        CUSOLVER_STATUS_IRS_NOT_SUPPORTED=21,
        CUSOLVER_STATUS_IRS_OUT_OF_RANGE=22,
        CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES=23,
        CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED=25,
        CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED=26,
        CUSOLVER_STATUS_IRS_MATRIX_SINGULAR=30,
        CUSOLVER_STATUS_INVALID_WORKSPACE=31
    }

    pub enum Struct_cusolverDnContext { }
    pub type cusolverDnHandle_t = *mut Struct_cusolverDnContext;

    #[derive(PartialEq, Debug, Clone, Copy)]
    #[repr(u32)]
    pub enum cusolverEigMode_t {
        CUSOLVER_EIG_MODE_NOVECTOR=0,
        CUSOLVER_EIG_MODE_VECTOR=1
    }

    #[derive(PartialEq, Debug, Clone, Copy)]
    #[repr(u32)]
    pub enum cusolverEigRange_t {
        CUSOLVER_EIG_RANGE_ALL=1001,
        CUSOLVER_EIG_RANGE_I=1002,
        CUSOLVER_EIG_RANGE_V=1003,
    }

    #[link(name = "cusolver")]
    extern "C" {
        pub fn cusolverDnCreate(
            handle: *mut cusolverDnHandle_t
        ) -> cusolverStatus_t;

        pub fn cusolverDnSsyevdx_bufferSize(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            range: cusolverEigRange_t,
            uplo: cublasFillMode_t,
            n: ::libc::c_int,
            A: *const ::libc::c_float,
            lda: ::libc::c_int,
            vl: ::libc::c_float,
            vu: ::libc::c_float,
            il: ::libc::c_int,
            iu: ::libc::c_int,
            meig: *mut ::libc::c_int,
            W: *const ::libc::c_float,
            lwork: *mut ::libc::c_int
        ) -> cusolverStatus_t;

        pub fn cusolverDnSsyevdx(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            range: cusolverEigRange_t,
            uplo: cublasFillMode_t,
            n: ::libc::c_int,
            A: *mut ::libc::c_float,
            lda: ::libc::c_int,
            vl: ::libc::c_float,
            vu: ::libc::c_float,
            il: ::libc::c_int,
            iu: ::libc::c_int,
            meig: *mut ::libc::c_int,
            W: *mut ::libc::c_float,
            work: *mut ::libc::c_float,
            lwork: ::libc::c_int,
            info: *mut ::libc::c_int
        ) -> cusolverStatus_t;
    }
}
