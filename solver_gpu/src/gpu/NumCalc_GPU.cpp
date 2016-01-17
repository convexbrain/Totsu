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


NCVec_GPU NumCalc_GPU::nullVec;


NumCalc_GPU::NumCalc_GPU()
{
	m_fatalErr = 0;

	int devNum = 0;

	if (cudaSetDevice(devNum)) m_fatalErr = __LINE__;
	if (cudaGetDeviceProperties(&m_cuDevProp, devNum)) m_fatalErr = __LINE__;

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

int NumCalc_GPU::calcSearchDir(NCMat_GPU &kkt, NCVec_GPU &rtDy)
{
	// Solve kkt * Dy = -r_t. Dy overwrites rtDy.
	// Also kkt will be modified by QR factorization.

	if (m_fatalErr) return m_fatalErr;

	NC_uint n = kkt.nRows();
	NC_uint lda = kkt.nRowsPitch();
	NC_uint ldc = rtDy.nRowsPitch();

	if (n != kkt.nCols()) return __LINE__;
	if (n != rtDy.nRows()) return __LINE__;

	int bufferSize = 0;
	if (cusolverDnDgeqrf_bufferSize(
		m_cuSolverDn,
		n, // rows of A
		n, // cols of A
		kkt.ptr(), // A
		lda, // pitch of A
		&bufferSize)) {
		return __LINE__;
	}

	m_solverBuf.realloc(bufferSize);
	m_solverTau.realloc(sizeof(NC_Scalar) * n);

	m_devInfo.realloc(sizeof(int));
	if (m_devInfo.setZero()) return __LINE__;

	// compute QR factorization: kkt -> Q * R
	if (cusolverDnDgeqrf(
		m_cuSolverDn,
		n, // rows of A
		n, // cols of A
		kkt.ptr(), // A
		lda, // pitch of A
		m_solverTau.ptr<NC_Scalar*>(),
		m_solverBuf.ptr<NC_Scalar*>(),
		bufferSize,
		m_devInfo.ptr<int*>()))
	{
		return __LINE__;
	}

	m_devInfo.copyToHost();
	int *pinfo = m_devInfo.hostPtr<int*>();
	if (0 != *pinfo) return __LINE__;

	// compute Q^T * rtDy
	if (cusolverDnDormqr(
		m_cuSolverDn,
		CUBLAS_SIDE_LEFT,
		CUBLAS_OP_T,
		n, // rows of A
		1, // cols of C??
		n, // cols of A??
		kkt.ptr(), // A
		lda, // pitch of A
		m_solverTau.ptr<NC_Scalar*>(),
		rtDy.ptr(), // C
		ldc, // pitch of C
		m_solverBuf.ptr<NC_Scalar*>(),
		bufferSize,
		m_devInfo.ptr<int*>()))
	{
		return __LINE__;
	}

	// compute R \ (Q^T * -rtDy)
	const NC_Scalar minusone = -1.0;
	if (cublasDtrsm(
		m_cuBlas,
		CUBLAS_SIDE_LEFT,
		CUBLAS_FILL_MODE_UPPER,
		CUBLAS_OP_N,
		CUBLAS_DIAG_NON_UNIT,
		n, // rows of B
		1, // cols of B
		&minusone,
		kkt.ptr(), // A
		lda, // pitch of A
		rtDy.ptr(), // B
		ldc)) // pitch of B
	{
		return __LINE__;
	}

	if (cudaDeviceSynchronize()) { // TODO
		return __LINE__;
	}

	return 0;
}

int NumCalc_GPU::calcAddKKT(NCMat_GPU &kkt, NCMat_GPU &B, bool Btr, NCVec_GPU &beta, NC_uint betaIdx)
{
	if (m_fatalErr) return m_fatalErr;

	const NC_Scalar one = 1.0;
	if (cublasDgeam(
		m_cuBlas,
		CUBLAS_OP_N,
		(Btr) ? CUBLAS_OP_T : CUBLAS_OP_N,
		kkt.nRows(),
		kkt.nCols(),
		&one,
		kkt.ptr(),
		kkt.nRowsPitch(),
		(beta.nRows() == 0) ? &one : beta.ptr() + betaIdx,
		B.ptr(),
		B.nRowsPitch(),
		kkt.ptr(),
		kkt.nRowsPitch()))
	{
		return __LINE__;
	}

	return 0;
}
