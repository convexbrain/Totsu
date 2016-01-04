/*
* This software contains source code provided by NVIDIA Corporation.
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

#include <cuda_runtime.h>

#include "cusolverDn.h"
#include "helper_cuda.h"

int linearSolverQR(
	cusolverDnHandle_t handle,
	int n,
	const double *Acopy,
	int lda,
	const double *b,
	double *x);

/***********************************************/

void gpu_calcSearchDirection(IPM_Matrix_IN kkt, IPM_Vector_IN r_t, IPM_Vector_IO Dy)
{
	cusolverDnHandle_t handle = NULL;
	cudaStream_t stream = NULL;

	int rowsA = 0; // number of rows of A
	int colsA = 0; // number of columns of A
	int lda = 0; // leading dimension in dense matrix

	double *h_A = NULL; // dense matrix from CSR(A)
	double *h_x = NULL; // a copy of d_x
	double *h_b = NULL; // b = ones(m,1)

	double *d_A = NULL; // a copy of h_A
	double *d_x = NULL; // x = A \ b
	double *d_b = NULL; // a copy of h_b

	lda = rowsA = colsA = int(kkt.rows());
	assert(kkt.rows() == kkt.cols());
	assert(kkt.cols() == r_t.size());
	assert(kkt.rows() == Dy.size());

	h_A = (double*)malloc(sizeof(double)*lda*colsA);
	h_x = (double*)malloc(sizeof(double)*colsA);
	h_b = (double*)malloc(sizeof(double)*rowsA);
	assert(NULL != h_A);
	assert(NULL != h_x);
	assert(NULL != h_b);

	memset(h_A, 0, sizeof(double)*lda*colsA);

	for (int row = 0; row < rowsA; row++)
	{
		for (int col = 0; col < colsA; col++)
		{
			h_A[row + col*lda] = kkt(row, col);
		}
		h_b[row] = -r_t(row);
	}

	checkCudaErrors(cusolverDnCreate(&handle));
	checkCudaErrors(cudaStreamCreate(&stream));

	checkCudaErrors(cusolverDnSetStream(handle, stream));


	checkCudaErrors(cudaMalloc((void **)&d_A, sizeof(double)*lda*colsA));
	checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(double)*colsA));
	checkCudaErrors(cudaMalloc((void **)&d_b, sizeof(double)*rowsA));

	checkCudaErrors(cudaMemcpy(d_A, h_A, sizeof(double)*lda*colsA, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_b, h_b, sizeof(double)*rowsA, cudaMemcpyHostToDevice));

	linearSolverQR(handle, rowsA, d_A, lda, d_b, d_x);

	checkCudaErrors(cudaMemcpy(h_x, d_x, sizeof(double)*colsA, cudaMemcpyDeviceToHost));

	for (int col = 0; col < colsA; col++)
	{
		Dy(col) = h_x[col];
	}

	if (handle) { checkCudaErrors(cusolverDnDestroy(handle)); }
	if (stream) { checkCudaErrors(cudaStreamDestroy(stream)); }

	if (h_A) { free(h_A); }
	if (h_x) { free(h_x); }
	if (h_b) { free(h_b); }

	if (d_A) { checkCudaErrors(cudaFree(d_A)); }
	if (d_x) { checkCudaErrors(cudaFree(d_x)); }
	if (d_b) { checkCudaErrors(cudaFree(d_b)); }

	cudaDeviceReset();
}
