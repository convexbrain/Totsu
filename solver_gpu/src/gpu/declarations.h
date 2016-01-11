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

#include "NumCalc_GPU.h"

void gpuwrap_calcSearchDir(NumCalc_GPU *pNC, IPM_Matrix_IN _kkt, IPM_Vector_IN r_t, IPM_Vector_IO Dy)
{
	const NC_uint n = NC_uint(_kkt.rows());

	NCMat_GPU kkt;
	NCVec_GPU rtDy;
	kkt.resize(n, n);
	rtDy.resize(n);

	const NC_uint pitch = kkt.nRowsPitch();

	NC_Scalar *pH_kkt = kkt.hostPtr();
	NC_Scalar *pH_rtDy = rtDy.hostPtr();

	for (NC_uint row = 0; row < n; row++)
	{
		for (NC_uint col = 0; col < n; col++)
		{
			pH_kkt[row + col * pitch] = _kkt(row, col);
		}
		pH_rtDy[row] = r_t(row);
	}

	kkt.copyToDevice();
	rtDy.copyToDevice();

	assert(pNC->calcSearchDir(kkt, rtDy) == 0);

	rtDy.copyToHost();

	for (NC_uint row = 0; row < n; row++)
	{
		Dy(row) = pH_rtDy[row];
	}
}

IPM_Scalar gpuwrap_calcMaxScaleBTLS(NumCalc_GPU *pNC, IPM_Vector_IN _lmd, IPM_Vector_IN _Dlmd)
{
	const NC_uint m = NC_uint(_lmd.rows());
	assert(m == _Dlmd.rows());

	NCVec_GPU lmd;
	NCVec_GPU Dlmd;
	lmd.resize(m);
	Dlmd.resize(m);

	NC_Scalar *pH_lmd = lmd.hostPtr();
	NC_Scalar *pH_Dlmd = Dlmd.hostPtr();

	for (NC_uint row = 0; row < m; row++) {
		pH_lmd[row] = _lmd(row);
		pH_Dlmd[row] = _Dlmd(row);
	}

	lmd.copyToDevice();
	Dlmd.copyToDevice();

	IPM_Scalar s_max;
	assert(pNC->calcMaxScaleBTLS(lmd, Dlmd, &s_max) == 0);

	return s_max;
}

