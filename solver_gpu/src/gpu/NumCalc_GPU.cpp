/*
* This software contains source code provided by NVIDIA Corporation.
*
*/
/*
* Copyright 2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#include "NumCalc_GPU.h"


NumCalc_GPU::NumCalc_GPU()
{
	m_fatalErr = 0;

	if (cusolverDnCreate(&m_cuSolverDn)) m_fatalErr = __LINE__;
	if (cudaStreamCreate(&m_cuStream)) m_fatalErr = __LINE__;
	if (cublasCreate(&m_cuBlas)) m_fatalErr = __LINE__;

	if (cusolverDnSetStream(m_cuSolverDn, m_cuStream)) m_fatalErr = __LINE__;
}

NumCalc_GPU::~NumCalc_GPU()
{
	if (m_cuSolverDn) cusolverDnDestroy(m_cuSolverDn);
	if (m_cuStream) cudaStreamDestroy(m_cuStream);
	if (m_cuBlas) cublasDestroy(m_cuBlas);
}

void NumCalc_GPU::resetDevice(void)
{
	cudaDeviceReset();
}

int NumCalc_GPU::calcSearchDirection(NCMat_GPU &kkt, NCVec_GPU &rtDy)
{
	if (m_fatalErr) return m_fatalErr;

	NC_uint n = kkt.m_nRows;
	NC_uint lda = kkt.m_nRowsPitch;

	if (n != kkt.m_nCols) return __LINE__;
	if (n != rtDy.m_nRows) return __LINE__;

	int bufferSize = 0;
	if (cusolverDnDgeqrf_bufferSize(m_cuSolverDn, n, n, kkt.ptr<NC_SCALAR*>(), lda, &bufferSize)) {
		return __LINE__;
	}

	m_solverBuf.realloc(bufferSize);
	m_solverTau.realloc(sizeof(NC_SCALAR) * n);

	m_devInfo.realloc(sizeof(int));
	if (m_devInfo.setZero()) return __LINE__;

	// compute QR factorization
	if (cusolverDnDgeqrf(m_cuSolverDn,
		n, n, kkt.ptr<NC_SCALAR*>(), lda,
		m_solverTau.ptr<NC_SCALAR*>(), m_solverBuf.ptr<NC_SCALAR*>(),
		bufferSize, m_devInfo.ptr<int*>()))
	{
		return __LINE__;
	}

	m_devInfo.copyToHost();
	int *pinfo = m_devInfo.hostPtr<int*>();
	if (0 != *pinfo) return __LINE__;

	// compute Q^T*b
	if (cusolverDnDormqr(
		m_cuSolverDn,
		CUBLAS_SIDE_LEFT,
		CUBLAS_OP_T,
		n,
		1,
		n,
		kkt.ptr<NC_SCALAR*>(),
		lda,
		m_solverTau.ptr<NC_SCALAR*>(),
		rtDy.ptr<NC_SCALAR*>(),
		n,
		m_solverBuf.ptr<NC_SCALAR*>(),
		bufferSize,
		m_devInfo.ptr<int*>()))
	{
		return __LINE__;
	}

	// x = R \ Q^T*b
	const NC_SCALAR minusone = -1.0;
	if (cublasDtrsm(
		m_cuBlas,
		CUBLAS_SIDE_LEFT,
		CUBLAS_FILL_MODE_UPPER,
		CUBLAS_OP_N,
		CUBLAS_DIAG_NON_UNIT,
		n,
		1,
		&minusone,
		kkt.ptr<NC_SCALAR*>(),
		lda,
		rtDy.ptr<NC_SCALAR*>(),
		n))
	{
		return __LINE__;
	}

	if (cudaDeviceSynchronize()) { // TODO
		return __LINE__;
	}

	return 0;
}
