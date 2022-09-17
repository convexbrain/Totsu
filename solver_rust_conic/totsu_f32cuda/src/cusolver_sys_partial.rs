#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use cublas_sys::cublasFillMode_t;

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
