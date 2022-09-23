use totsu_core::solver::{SliceLike, LinAlg};
use totsu_core::LinAlgEx;
use rustacuda::prelude::*;
use cublas_sys::*;
use crate::cuda_mgr;
use crate::f32cuda_slice::F32CUDASlice;
use crate::cusolver_sys_partial::*;

//

/// `f32`-specific [`LinAlgEx`] implementation using `rustacuda`, `cublas-sys` and [`crate::cusolver_sys_partial`].
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
/// 
/// You need a [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) to link.
#[derive(Clone)]
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

    fn map_eig_worklen(n: usize) -> usize
    {
        let len_a = n * n;

        len_a + eig_func_worklen(n)
    }

    fn map_eig<M>(mat: &mut F32CUDASlice, scale_diag: Option<f32>, _eps_zero: f32, work: &mut F32CUDASlice, map: M)
    where M: Fn(f32)->Option<f32>
    {
        let sn = mat.len();

        let n = (((8 * sn + 1) as f64).sqrt() as usize - 1) / 2;
        assert_eq!(n * (n + 1) / 2, sn);

        assert!(work.len() >= Self::map_eig_worklen(n));

        let (mut a, mut w_z_work) = work.split_mut(n * n);

        vec_to_mat(mat, &mut a, scale_diag);
    
        eig_func(&mut a, n, &mut w_z_work, map);

        mat_to_vec(&mut a, mat, scale_diag);
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
            n as i32, std::ptr::null(), n as i32,
            0., f32::INFINITY, 0, 0,
            std::ptr::null_mut(), std::ptr::null(),
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
    let (mut w, mut z) = w_z_work.split_mut(n);
    let (mut z, mut work) = z.split_mut(n * n);
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
    
    let w_ref = w.get_ref();
    for i in 0.. meig as usize {
        if let Some(e) = func(w_ref[i]) {
            let (_, ref_z) = z.split_ref(i * n);

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

fn vec_to_mat(v: &F32CUDASlice, m: &mut F32CUDASlice, scale: Option<f32>)
{
    let l = v.len();
    let n = (m.len() as f64).sqrt() as usize;

    assert_eq!(m.len(), n * n);
    assert_eq!(n * (n + 1) / 2, l);

    // The vector is a symmetric matrix, packing the upper-triangle by columns.
    let mut iv = 0;
    for c in 0.. n {
        let (_, mut spl_m) = m.split_mut(c * n);
        let (mut col_m, _) = spl_m.split_mut(c + 1);

        let (_, spl_v) = v.split_ref(iv);
        let (col_v, _) = spl_v.split_ref(c + 1);
        iv += c + 1;
        F32CUDA::copy(&col_v, &mut col_m);
    }

    if let Some(scl) = scale {
        // scale diagonals
        unsafe {
            let st = cublasSscal_v2(
                cuda_mgr::cublas_handle(),
                n as i32,
                &scl, m.get_dev_mut().as_mut_ptr(), (n + 1) as i32
            );
            assert_eq!(st, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        }
    }
}

fn mat_to_vec(m: &mut F32CUDASlice, v: &mut F32CUDASlice, scale: Option<f32>)
{
    let l = v.len();
    let n = (m.len() as f64).sqrt() as usize;

    assert_eq!(m.len(), n * n);
    assert_eq!(n * (n + 1) / 2, l);

    if let Some(scl) = scale {
        // revert scaled diagonals
        unsafe {
            let st = cublasSscal_v2(
                cuda_mgr::cublas_handle(),
                n as i32,
                &scl.recip(), m.get_dev_mut().as_mut_ptr(), (n + 1) as i32
            );
            assert_eq!(st, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        }
    }

    // The vector is a symmetric matrix, packing the upper-triangle by columns.
    let mut iv = 0;
    for c in 0.. n {
        let (_, spl_m) = m.split_ref(c * n);
        let (col_m, _) = spl_m.split_ref(c + 1);

        let (_, mut spl_v) = v.split_mut(iv);
        let (mut col_v, _) = spl_v.split_mut(c + 1);
        iv += c + 1;
        F32CUDA::copy(&col_m, &mut col_v);
    }
}
