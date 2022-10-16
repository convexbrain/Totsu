//! CUDA manager module.

use std::prelude::v1::*;
use std::thread_local;
use std::rc::Rc;
use std::ffi::CString;
use num_traits::Zero;
use rustacuda::prelude::*;
use rustacuda::memory::{DeviceBuffer, DeviceCopy};
use cublas_sys::*;
use super::cusolver_sys_partial::*;

pub struct CudaHandle
{
    pub context: Context,
    pub module: Module,
    pub stream: Stream,
}

struct CudaManager
{
    cuda_handle: Rc<CudaHandle>,
    cublas_handle: cublasHandle_t,
    cusolver_handle: cusolverDnHandle_t,
}

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
        let context = Context::create_and_push(
            ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
            device
        );
        if context.is_err() {
            log::error!("CUDA context failed to create");
        }
        let context = context.unwrap();

        // Load the kernel module
        let module_data = CString::new(include_str!("../kernel/kernel.ptx")).unwrap();
        let module = Module::load_from_string(&module_data);
        if module.is_err() {
            log::error!("CUDA module failed to create");
        }
        let module = module.unwrap();

        // Create a stream
        let stream = Stream::new(StreamFlags::DEFAULT, None);
        if stream.is_err() {
            log::error!("CUDA stream failed to create");
        }
        let stream = stream.unwrap();

        let cuda_handle = Rc::new(CudaHandle {
            context,
            module,
            stream,
        });

        // cuBLAS handle
        let mut cublas_handle: cublasHandle_t = std::ptr::null_mut();
        unsafe {
            let st = cublasCreate_v2(&mut cublas_handle);
            if st != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                log::error!("cuBLAS handle failed to create");
            }
            assert_eq!(st, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        }

        // cuSOLVER handle
        let mut cusolver_handle: cusolverDnHandle_t = std::ptr::null_mut();
        unsafe {
            let st = cusolverDnCreate(&mut cusolver_handle);
            if st != cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
                log::error!("cuSOLVER handle failed to create");
            }
            assert_eq!(st, cusolverStatus_t::CUSOLVER_STATUS_SUCCESS);
        }

        log::debug!("CUDA_MANAGER created");
        CudaManager {
            cuda_handle,
            cublas_handle,
            cusolver_handle,
        }
    }
}

impl Drop for CudaManager
{
    fn drop(&mut self)
    {
        unsafe {
            let st = cublasDestroy_v2(self.cublas_handle);
            if st != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                log::error!("cuBLAS handle failed to destroy");
            }
            //assert_eq!(st, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        }

        unsafe {
            let st = cusolverDnDestroy(self.cusolver_handle);
            if st != cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
                log::error!("cuSOLVER handle failed to destroy");
            }
            //assert_eq!(st, cusolverStatus_t::CUSOLVER_STATUS_SUCCESS);
        }

        log::debug!("CUDA_MANAGER dropped");
    }
}

//

thread_local!(static CUDA_MANAGER: CudaManager = CudaManager::new());

/// Allocates a new device buffer of the same contents as a given slice.
/// 
/// Returns the device buffer.
/// * `s` is the original slice.
pub fn buf_from_slice<T>(s: &[T]) -> DeviceBuffer<T>
where T: DeviceCopy
{
    CUDA_MANAGER.with(|_| { // ensure that rustacuda::init is done
        DeviceBuffer::<T>::from_slice(s).unwrap()
    })
}

/// Allocates a new device buffer with zeroes.
/// 
/// Returns the device buffer with a `length` of the slice.
pub fn buf_zeroes<T>(length: usize) -> DeviceBuffer<T>
where T: Zero
{
    CUDA_MANAGER.with(|_| { // ensure that rustacuda::init is done
        unsafe {
            DeviceBuffer::zeroed(length).unwrap()
        }
    })
}

/// Gets the CUDA handle.
pub fn cuda_handle() -> Rc<CudaHandle>
{
    CUDA_MANAGER.with(|mgr| {
        mgr.cuda_handle.clone()
    })
}

/// Gets the cuBLAS handle.
pub fn cublas_handle() -> cublasHandle_t
{
    CUDA_MANAGER.with(|mgr| {
        mgr.cublas_handle
    })
}

/// Gets the cuSOLVER handle.
pub fn cusolver_handle() -> cusolverDnHandle_t
{
    CUDA_MANAGER.with(|mgr| {
        mgr.cusolver_handle
    })
}
