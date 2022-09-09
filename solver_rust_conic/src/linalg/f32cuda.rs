use std::vec::Vec;
use std::vec;
use std::collections::HashMap;
use std::sync::{Mutex, Arc};
use std::boxed::Box;
use std::pin::Pin;
use super::{SliceRef, SliceMut, SliceLike, LinAlg, LinAlgEx};
use rustacuda::prelude::*;
use rustacuda::memory::{DeviceBuffer, DeviceSlice};
use cublas_sys::*;
use cusolver_sys_partial::*;
use once_cell::sync::Lazy;

//

pub mod cuda_mgr
{
    use std::thread_local;
    use std::rc::Rc;
    use rustacuda::prelude::*;
    use rustacuda::memory::DeviceBuffer;
    use cublas_sys::*;
    use super::cusolver_sys_partial::*;

    struct CudaManager
    {
        cuda_ctx: Rc<Context>,
        cublas_handle: cublasHandle_t,
        cusolver_handle: cusolverDnHandle_t,
    }

    unsafe impl Send for CudaManager {}
    unsafe impl Sync for CudaManager {}

    impl CudaManager
    {
        fn new() -> CudaManager
        {
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
            let cuda_ctx = Rc::new(cuda_ctx.unwrap());

            // cuBLAS handle
            let mut cublas_handle: cublasHandle_t = core::ptr::null_mut();
            unsafe {
                let st = cublasCreate_v2(&mut cublas_handle);
                if st != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    log::error!("cuBLAS handle failed to create");
                }
                assert_eq!(st, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
            }

            // cuSOLVER handle
            let mut cusolver_handle: cusolverDnHandle_t = core::ptr::null_mut();
            unsafe {
                let st = cusolverDnCreate(&mut cusolver_handle);
                if st != cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
                    log::error!("cuSOLVER handle failed to create");
                }
                assert_eq!(st, cusolverStatus_t::CUSOLVER_STATUS_SUCCESS);
            }

            log::debug!("CUDA_MANAGER created");
            CudaManager {
                cuda_ctx,
                cublas_handle,
                cusolver_handle,
            }
        }
    }

    thread_local!(static CUDA_MANAGER: CudaManager = CudaManager::new());

    /// TODO: doc
    pub fn buf_from_slice(s: &[f32]) -> DeviceBuffer<f32>
    {
        CUDA_MANAGER.with(|_| { // ensure that rustacuda::init is done
            DeviceBuffer::from_slice(s).unwrap()
        })
    }

    /// TODO: doc
    pub fn buf_zeroes<T>(length: usize) -> DeviceBuffer<T>
    {
        CUDA_MANAGER.with(|_| { // ensure that rustacuda::init is done
            unsafe {
                DeviceBuffer::zeroed(length).unwrap()
            }
        })
    }

    /// TODO: doc
    pub fn context() -> Rc<Context>
    {
        CUDA_MANAGER.with(|mgr| {
            mgr.cuda_ctx.clone()
        })
    }

    /// TODO: doc
    pub fn cublas_handle() -> cublasHandle_t
    {
        CUDA_MANAGER.with(|mgr| {
            mgr.cublas_handle
        })
    }

    /// TODO: doc
    pub fn cusolver_handle() -> cusolverDnHandle_t
    {
        CUDA_MANAGER.with(|mgr| {
            mgr.cusolver_handle
        })
    }
}

//

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

// TODO: mod
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
            dev_buf: Arc::new(Mutex::new(F32CUDABuf(cuda_mgr::buf_from_slice(s)))),
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

    fn new_zeroes<'a>(&mut self, length: usize) -> &'a mut F32CUDASlice
    {
        let idx = self.cnt;
        self.cnt = idx + 1;

        let cs = F32CUDASlice {
            idx,
            parent_idx: None,
            dev_buf: Arc::new(Mutex::new(F32CUDABuf(cuda_mgr::buf_zeroes(length)))),
            host_buf: Arc::new(Mutex::new(vec![0.; length])),
            sta: 0,
            end: length,
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
            mutator: Mutex::new(*cso.mutator.try_lock().unwrap()),
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
        let cs_mut = *cs.mutator.try_lock().unwrap();
        let cs_par_idx = cs.parent_idx;
        let mut cs_par_mut = None;

        if let Some(par_idx) = cs_par_idx {
            let par = self.map.get_mut(&par_idx).unwrap();
            let mut par_mut = par.mutator.try_lock().unwrap();

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
                    if *cs.mutator.try_lock().unwrap() == CUDASliceMut::Dev {
                        let db = &cs.dev_buf.try_lock().unwrap().0[cs.sta..cs.end];
                        let mut hb = cs.host_buf.try_lock().unwrap();
                        let mut hb_mut = &mut hb[cs.sta..cs.end];
                        
                        db.copy_to(&mut hb_mut).unwrap();
                    }
                },
                CUDASliceMut::Dev => {
                    if *cs.mutator.try_lock().unwrap() == CUDASliceMut::Host {
                        let db = &mut cs.dev_buf.try_lock().unwrap().0[cs.sta..cs.end];
                        let mut hb = cs.host_buf.try_lock().unwrap();
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

// TODO: try thread_local
// TODO: encapsulate try_lock
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
        let mut mgr = SLICE_MANAGER.try_lock().unwrap();
        let cs = mgr.new_slice(s);
        
        //std::println!("{} new", cs.idx);
        SliceRef {s: cs}
    }

    fn new_mut(s: &mut[f32]) -> SliceMut<'_, F32CUDASlice>
    {
        let mut mgr = SLICE_MANAGER.try_lock().unwrap();
        let cs = mgr.new_slice(s);

        //std::println!("{} new_mut", cs.idx);
        SliceMut {s: cs}
    }
    
    fn split_at(&self, mid: usize) -> (SliceRef<'_, F32CUDASlice>, SliceRef<'_, F32CUDASlice>)
    {
        let mut mgr = SLICE_MANAGER.try_lock().unwrap();
        let cs0 = mgr.split_slice(self, 0, mid);
        let cs1 = mgr.split_slice(self, mid, self.len());

        //std::println!("{} split from {}", cs0.idx, self.idx);
        //std::println!("{} split from {}", cs1.idx, self.idx);
        (SliceRef {s: cs0}, SliceRef {s: cs1})
    }

    fn split_at_mut(&mut self, mid: usize) -> (SliceMut<'_, F32CUDASlice>, SliceMut<'_, F32CUDASlice>)
    {
        let mut mgr = SLICE_MANAGER.try_lock().unwrap();
        let cs0 = mgr.split_slice(self, 0, mid);
        let cs1 = mgr.split_slice(self, mid, self.len());

        //std::println!("{} split_mut from {}", cs0.idx, self.idx);
        //std::println!("{} split_mut from {}", cs1.idx, self.idx);
        (SliceMut {s: cs0}, SliceMut {s: cs1})
    }

    fn drop(&self)
    {
        //std::println!("{} drop", self.idx);
        let mut mgr = SLICE_MANAGER.try_lock().unwrap();
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
        let mut mutator = self.mutator.try_lock().unwrap();
        //std::println!("  {:?} mutator", *mutator);
        match *mutator {
            CUDASliceMut::Sync | CUDASliceMut::Host => {},
            CUDASliceMut::Dev => {
                let db = &self.dev_buf.try_lock().unwrap().0[self.sta..self.end];
                let mut hb = &mut self.host_buf.try_lock().unwrap()[self.sta..self.end];
                
                db.copy_to(&mut hb).unwrap();

                *mutator = CUDASliceMut::Sync;
            },
        }

        let hb_ref = &self.host_buf.try_lock().unwrap()[self.sta..self.end];

        unsafe {
            core::mem::transmute::<&[f32], &[f32]>(hb_ref)
        }
    }

    fn get_mut(&mut self) -> &mut[f32]
    {
        //std::println!("  {} get_mut", self.idx);
        let mut mutator = self.mutator.try_lock().unwrap();
        //std::println!("    {:?} mutator", *mutator);
        match *mutator {
            CUDASliceMut::Sync => {
                *mutator = CUDASliceMut::Host;
            },
            CUDASliceMut::Host => {},
            CUDASliceMut::Dev => {
                let db = &self.dev_buf.try_lock().unwrap().0[self.sta..self.end];
                let mut hb = &mut self.host_buf.try_lock().unwrap()[self.sta..self.end];

                db.copy_to(&mut hb).unwrap();

                *mutator = CUDASliceMut::Host;
            },
        }

        let hb_mut = &mut self.host_buf.try_lock().unwrap()[self.sta..self.end];

        unsafe {
            core::mem::transmute::<&mut[f32], &mut[f32]>(hb_mut)
        }
    }
}

impl F32CUDASlice
{
    /// TODO: doc
    pub fn new_zeroes(length: usize) -> SliceMut<'static, F32CUDASlice>
    {
        let mut mgr = SLICE_MANAGER.try_lock().unwrap();
        let cs = mgr.new_zeroes(length);

        SliceMut {s: cs}
    }    

    /// TODO: doc
    pub fn get_dev(&self) -> &DeviceSlice<f32>
    {
        let mut mutator = self.mutator.try_lock().unwrap();
        match *mutator {
            CUDASliceMut::Sync | CUDASliceMut::Dev => {},
            CUDASliceMut::Host => {
                let db = &mut self.dev_buf.try_lock().unwrap().0[self.sta..self.end];
                let mut hb = &mut self.host_buf.try_lock().unwrap()[self.sta..self.end];
                
                db.copy_from(&mut hb).unwrap();

                *mutator = CUDASliceMut::Sync;
            },
        }

        let db_ref = &self.dev_buf.try_lock().unwrap().0[self.sta..self.end];

        unsafe {
            core::mem::transmute::<&DeviceSlice<f32>, &DeviceSlice<f32>>(db_ref)
        }
    }

    /// TODO: doc
    pub fn get_dev_mut(&mut self) -> &mut DeviceSlice<f32>
    {
        let mut mutator = self.mutator.try_lock().unwrap();
        match *mutator {
            CUDASliceMut::Sync => {
                *mutator = CUDASliceMut::Dev;
            },
            CUDASliceMut::Dev => {},
            CUDASliceMut::Host => {
                let db = &mut self.dev_buf.try_lock().unwrap().0[self.sta..self.end];
                let mut hb = &mut self.host_buf.try_lock().unwrap()[self.sta..self.end];
                
                db.copy_from(&mut hb).unwrap();

                *mutator = CUDASliceMut::Dev;
            },
        }

        let db_mut = &mut self.dev_buf.try_lock().unwrap().0[self.sta..self.end];

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
                cuda_mgr::cublas_handle(),
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
                cuda_mgr::cublas_handle(),
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
                cuda_mgr::cublas_handle(),
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
                cuda_mgr::cublas_handle(),
                x.len() as i32,
                &alpha, x.get_dev().as_ptr(), 1,
                y.get_dev_mut().as_mut_ptr(), 1
            );
            assert_eq!(st, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        }
    }

    fn adds(s: f32, y: &mut F32CUDASlice)
    {
        let one = cuda_mgr::buf_from_slice(&[1.]);

        unsafe {
            let st = cublasSaxpy_v2(
                cuda_mgr::cublas_handle(),
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
                    cuda_mgr::cublas_handle(),
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
                cuda_mgr::cublas_handle(),
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
                cuda_mgr::cublas_handle(),
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
                cuda_mgr::cublas_handle(),
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
        let n = (((8 * sn + 1) as f64).sqrt() as usize - 1) / 2;
        assert_eq!(n * (n + 1) / 2, sn);

        let len_a = n * n;

        len_a + eig_func_worklen(n)
    }

    fn proj_psd(x: &mut F32CUDASlice, _eps_zero: f32, work: &mut F32CUDASlice)
    {
        let sn = x.len();

        let n = (((8 * sn + 1) as f64).sqrt() as usize - 1) / 2;
        assert_eq!(n * (n + 1) / 2, sn);
        assert!(work.len() >= Self::proj_psd_worklen(sn));

        let (mut a, mut w_z_work) = work.split_at_mut(n * n);

        vec_to_mat(x, &mut a, true);
    
        eig_func(&mut a, n, &mut w_z_work, |e| {
            if e > 0. {
                Some(e)
            }
            else {
                None
            }
        });

        mat_to_vec(&mut a, x, true);
    }

    fn sqrt_spmat_worklen(n: usize) -> usize
    {
        let len_a = n * n;

        len_a + eig_func_worklen(n)
    }

    fn sqrt_spmat(mat: &mut F32CUDASlice, _eps_zero: f32, work: &mut F32CUDASlice)
    {
        // TODO: cuda test
        /*
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
        */
        let sn = mat.len();

        let n = (((8 * sn + 1) as f64).sqrt() as usize - 1) / 2;
        assert_eq!(n * (n + 1) / 2, sn);
        assert!(work.len() >= Self::proj_psd_worklen(sn));

        let (mut a, mut w_z_work) = work.split_at_mut(n * n);

        vec_to_mat(mat, &mut a, false);
    
        eig_func(&mut a, n, &mut w_z_work, |e| {
            if e > 0. {
                Some(e.sqrt())
            }
            else {
                None
            }
        });

        mat_to_vec(&mut a, mat, false);
    }
}

//

fn eig_func_worklen(n: usize) -> usize
{
    let mut lwork: i32 = 0;
    
    unsafe {
        let st = cusolverDnSsyevdx_bufferSize(
            cuda_mgr::cusolver_handle(),
            cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR,
            cusolverEigRange_t::CUSOLVER_EIG_RANGE_V,
            cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
            n as i32, core::ptr::null(), n as i32,
            0., f32::INFINITY, 0, 0,
            core::ptr::null_mut(), core::ptr::null(),
            &mut lwork
        );
        assert_eq!(st, cusolverStatus_t::CUSOLVER_STATUS_SUCCESS);
    }

    let len_w = n;
    let len_z = n * n;

    len_w + len_z + lwork as usize
}

fn eig_func<E>(a: &mut F32CUDASlice, n: usize, w_z_work: &mut F32CUDASlice, func: E)
where E: Fn(f32)->Option<f32>
{
    let (mut w, mut z) = w_z_work.split_at_mut(n);
    let (mut z, mut work) = z.split_at_mut(n * n);
    let lwork = eig_func_worklen(n) - n - n * n;

    let mut meig: i32 = 0;
    let mut dev_info = cuda_mgr::buf_zeroes(1);

    unsafe {
        let st = cusolverDnSsyevdx(
            cuda_mgr::cusolver_handle(),
            cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR,
            cusolverEigRange_t::CUSOLVER_EIG_RANGE_V,
            cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
            n as i32, a.get_dev_mut().as_mut_ptr(), n as i32,
            0., f32::INFINITY, 0, 0,
            &mut meig, w.get_dev_mut().as_mut_ptr(),
            work.get_dev_mut().as_mut_ptr(), lwork as i32,
            dev_info.as_mut_ptr()
        );
        assert_eq!(st, cusolverStatus_t::CUSOLVER_STATUS_SUCCESS);
    }

    let mut info = [0];
    dev_info.copy_to(&mut info).unwrap();
    assert_eq!(info[0], 0);

    //

    a.get_dev().copy_to(z.get_dev_mut()).unwrap();

    let alpha = 0.;
    unsafe {
        let st = cublasSscal_v2(
            cuda_mgr::cublas_handle(),
            (n * n) as i32,
            &alpha, a.get_dev_mut().as_mut_ptr(), 1
        );
        assert_eq!(st, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    }

    //
    
    for i in 0.. meig as usize {
        if let Some(e) = func(w.get()[i]) {
            let (_, ref_z) = z.split_at(i * n);

            unsafe {
                let st = cublasSsyr_v2(
                    cuda_mgr::cublas_handle(),
                    cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
                    n as i32,
                    &e, ref_z.get_dev().as_ptr(), 1,
                    a.get_dev_mut().as_mut_ptr(), n as i32
                );
                assert_eq!(st, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
            }
        }
    }
}

fn vec_to_mat(v: &F32CUDASlice, m: &mut F32CUDASlice, scale: bool)
{
    let l = v.len();
    let n = (m.len() as f64).sqrt() as usize;

    assert_eq!(m.len(), n * n);
    assert_eq!(n * (n + 1) / 2, l);

    // The vector is a symmetric matrix, packing the upper-triangle by columns.
    let mut iv = 0;
    for c in 0.. n {
        let (_, mut spl_m) = m.split_at_mut(c * n);
        let (mut col_m, _) = spl_m.split_at_mut(c + 1);

        let (_, spl_v) = v.split_at(iv);
        let (col_v, _) = spl_v.split_at(c + 1);
        iv += c + 1;
        F32CUDA::copy(&col_v, &mut col_m);
    }

    if scale {
        // scale diagonals to match the resulted matrix norm with the vector norm multiplied by 2
        unsafe {
            let st = cublasSscal_v2(
                cuda_mgr::cublas_handle(),
                n as i32,
                &2_f32.sqrt(), m.get_dev_mut().as_mut_ptr(), (n + 1) as i32
            );
            assert_eq!(st, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        }
    }
}

fn mat_to_vec(m: &mut F32CUDASlice, v: &mut F32CUDASlice, scale: bool)
{
    let l = v.len();
    let n = (m.len() as f64).sqrt() as usize;

    assert_eq!(m.len(), n * n);
    assert_eq!(n * (n + 1) / 2, l);

    if scale {
        // scale diagonals to match the resulted vector norm with the matrix norm multiplied by 0.5
        unsafe {
            let st = cublasSscal_v2(
                cuda_mgr::cublas_handle(),
                n as i32,
                &0.5_f32.sqrt(), m.get_dev_mut().as_mut_ptr(), (n + 1) as i32
            );
            assert_eq!(st, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        }
    }

    // The vector is a symmetric matrix, packing the upper-triangle by columns.
    let mut iv = 0;
    for c in 0.. n {
        let (_, spl_m) = m.split_at(c * n);
        let (col_m, _) = spl_m.split_at(c + 1);

        let (_, mut spl_v) = v.split_at_mut(iv);
        let (mut col_v, _) = spl_v.split_at_mut(c + 1);
        iv += c + 1;
        F32CUDA::copy(&col_m, &mut col_v);
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
