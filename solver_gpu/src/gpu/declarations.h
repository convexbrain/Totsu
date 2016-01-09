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

void gpuwrap_calcSearchDirection(NumCalc_GPU *pNC, IPM_Matrix_IN _kkt, IPM_Vector_IN r_t, IPM_Vector_IO Dy)
{
	const NC_uint n = NC_uint(_kkt.rows());
	NCMat_GPU kkt(n, n, n, n);
	NCVec_GPU rtDy(n, n);

	NC_SCALAR *pH_kkt = kkt.hostPtr<NC_SCALAR*>();
	NC_SCALAR *pH_rtDy = rtDy.hostPtr<NC_SCALAR*>();

	for (NC_uint row = 0; row < n; row++)
	{
		for (NC_uint col = 0; col < n; col++)
		{
			pH_kkt[row + col * n] = _kkt(row, col);
		}
		pH_rtDy[row] = r_t(row);
	}

	kkt.copyToDevice();
	rtDy.copyToDevice();

	assert(pNC->calcSearchDirection(kkt, rtDy) == 0);

	rtDy.copyToHost();

	for (NC_uint row = 0; row < n; row++)
	{
		Dy(row) = pH_rtDy[row];
	}
}
