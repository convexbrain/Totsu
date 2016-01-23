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

#include <device_launch_parameters.h>

#ifdef __CUDACC__
#define KPRMS2(grid, block) <<< grid, block >>>
#define KPRMS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KPRMS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KPRMS2(grid, block)
#define KPRMS3(grid, block, sh_mem)
#define KPRMS4(grid, block, sh_mem, stream)
#endif


__global__ void
kernel_calcMaxScaleBTLS(const NC_Scalar *lmd, const NC_Scalar *Dlmd, NC_uint m, NC_Scalar *pSclMax)
{
	// TODO: parallelize

	int tid = threadIdx.x;

	NC_Scalar s_max = 1.0;
	for (NC_uint i = 0; i < m; i++)
	{
		if (Dlmd[i] < -NC_SCALAR_MIN) // to avoid zero-division by Dlmd
		{
			s_max = fmin(s_max, -lmd[i] / Dlmd[i]);
		}
	}

	if (tid == 0) {
		pSclMax[0] = s_max;
	}
}

int NumCalc_GPU::calcMaxScaleBTLS(NCVec_GPU &lmd, NCVec_GPU &Dlmd, NC_Scalar *pSclMax)
{
	if (m_fatalErr) return m_fatalErr;

	m_sMax.realloc(sizeof(NC_Scalar));

	kernel_calcMaxScaleBTLS KPRMS2(1, 1) (
		lmd.ptr(),
		Dlmd.ptr(),
		lmd.nRows(),
		m_sMax.ptr<NC_Scalar*>()
		);

	m_sMax.copyToHost();

	*pSclMax = *(m_sMax.hostPtr<NC_Scalar*>());

	return 0; // TODO: error check
}

__global__ void
kernel_calcCentResidual(const NC_Scalar *lmd, const NC_Scalar *f_i, NC_Scalar inv_t, NC_uint m, NC_Scalar *r_cent)
{
	// TODO: parallelize
	// int tid = threadIdx.x;

	for (NC_uint i = 0; i < m; i++)
	{
		r_cent[i] = -lmd[i] * f_i[i] - inv_t * m;
	}
}

int NumCalc_GPU::calcCentResidual(NCVec_GPU &lmd, NCVec_GPU &f_i, NC_Scalar inv_t, NCVec_GPU &r_cent)
{
	if (m_fatalErr) return m_fatalErr;

	kernel_calcCentResidual KPRMS2(1, 1) (
		lmd.ptr(),
		f_i.ptr(),
		inv_t,
		lmd.nRows(),
		r_cent.ptr()
		);

	return 0; // TODO: error check
}

