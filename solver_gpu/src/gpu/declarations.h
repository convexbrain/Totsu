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


// TODO: should be held in PrimalDualIPM ?
static struct {
	NCVec_GPU y;
	// sub vector
	NCVec_GPU x;
	NCVec_GPU lmd;
	NCVec_GPU nu;

	NCMat_GPU kkt;
	// sub matrix
	NCMat_GPU kkt_x_dual;
	NCMat_GPU kkt_lmd_dual;
	NCMat_GPU kkt_nu_dual;
	NCMat_GPU kkt_x_cent;
	NCMat_GPU kkt_lmd_cent;
	NCMat_GPU kkt_x_pri;

	// tmp
	NCMat_GPU DDf;
	NCMat_GPU Df_i;
	NCVec_GPU f_i;
	NCMat_GPU A;
	NCVec_GPU rtDy;
	NCVec_GPU Dlmd;
} g;

void gpuwrap_clearKKT(NumCalc_GPU *pNC, IPM_uint n, IPM_uint m, IPM_uint p)
{
	g.kkt.resize(n + m + p, n + m + p);

	g.kkt.setZero();

	// sub matrix
	g.kkt_x_dual.sub(g.kkt, 0, 0, n, n);
	g.kkt_lmd_dual.sub(g.kkt, 0, n, n, m);
	g.kkt_nu_dual.sub(g.kkt, 0, n + m, n, p);
	g.kkt_x_cent.sub(g.kkt, n, 0, m, n);
	g.kkt_lmd_cent.sub(g.kkt, n, n, m, m);
	g.kkt_x_pri.sub(g.kkt, n + m, 0, p, n);
}

void gpuwrap_set_y(IPM_Vector_IN y, IPM_uint n, IPM_uint m, IPM_uint p)
{
	const NC_uint nmp = NC_uint(y.rows());

	g.y.resize(nmp);

	NC_Scalar *pH_y = g.y.hostPtr();

	for (NC_uint row = 0; row < nmp; row++)
	{
		pH_y[row] = y(row);
	}

	g.y.copyToDevice();

	// sub vectors
	g.x.sub(g.y, 0, n);
	g.lmd.sub(g.y, n, m);
	g.nu.sub(g.y, n + m, p);
}

void gpuwrap_addKKT_x_dual(NumCalc_GPU *pNC, IPM_Matrix_IN DDf, int idx)
{
	const NC_uint n = NC_uint(DDf.rows());
	assert(n == DDf.cols());

	g.DDf.resize(n, n);

	const NC_uint pitch = g.DDf.nRowsPitch();

	NC_Scalar *pH_DDf = g.DDf.hostPtr();

	for (NC_uint row = 0; row < n; row++)
	{
		for (NC_uint col = 0; col < n; col++)
		{
			pH_DDf[row + col * pitch] = DDf(row, col);
		}
	}

	g.DDf.copyToDevice();

	if (idx > 0) {
		assert(pNC->calcAddKKT(g.kkt_x_dual, g.DDf, false, g.lmd, idx) == 0); // TODO: error
	}
	else {
		assert(pNC->calcAddKKT(g.kkt_x_dual, g.DDf, false, NumCalc_GPU::nullVec, 0) == 0); // TODO: error
	}
}

void gpuwrap_setKKT_Df_i(NumCalc_GPU *pNC, IPM_Matrix_IN Df_i)
{
	const NC_uint m = NC_uint(Df_i.rows());
	const NC_uint n = NC_uint(Df_i.cols());

	g.Df_i.resize(m, n);

	const NC_uint pitch = g.Df_i.nRowsPitch();

	NC_Scalar *pH_Df_i = g.Df_i.hostPtr();

	for (NC_uint row = 0; row < m; row++)
	{
		for (NC_uint col = 0; col < n; col++)
		{
			pH_Df_i[row + col * pitch] = Df_i(row, col);
		}
	}

	g.Df_i.copyToDevice();

	assert(pNC->calcAddKKT(g.kkt_lmd_dual, g.Df_i, true, NumCalc_GPU::nullVec, 0) == 0); // TODO: error
	assert(pNC->calcMinusDiagMulKKT(g.kkt_x_cent, g.Df_i, g.lmd) == 0); // TODO: error
}

void gpuwrap_setKKT_f_i(NumCalc_GPU *pNC, IPM_Vector_IN f_i)
{
	const NC_uint m = NC_uint(f_i.rows());

	g.f_i.resize(m);

	NC_Scalar *pH_f_i = g.f_i.hostPtr();

	for (NC_uint row = 0; row < m; row++)
	{
		pH_f_i[row] = f_i(row);
	}

	g.f_i.copyToDevice();

	assert(pNC->calcMinusDiagKKT(g.kkt_lmd_cent, g.f_i) == 0); // TODO: error
}

void gpuwrap_setKKT_A(NumCalc_GPU *pNC, IPM_Matrix_IN A)
{
	const NC_uint p = NC_uint(A.rows());
	const NC_uint n = NC_uint(A.cols());

	g.A.resize(p, n);

	const NC_uint pitch = g.A.nRowsPitch();

	NC_Scalar *pH_A = g.A.hostPtr();

	for (NC_uint row = 0; row < p; row++)
	{
		for (NC_uint col = 0; col < n; col++)
		{
			pH_A[row + col * pitch] = A(row, col);
		}
	}

	g.A.copyToDevice();

	assert(pNC->calcAddKKT(g.kkt_nu_dual, g.A, true, NumCalc_GPU::nullVec, 0) == 0); // TODO: error
	assert(pNC->calcAddKKT(g.kkt_x_pri, g.A, false, NumCalc_GPU::nullVec, 0) == 0); // TODO: error
}

void gpuwrap_calcSearchDir(NumCalc_GPU *pNC, IPM_Vector_IN r_t, IPM_Vector_IO Dy)
{
	const NC_uint nmp = NC_uint(r_t.rows());
	assert(nmp == NC_uint(Dy.rows()));

	g.rtDy.resize(nmp);

	NC_Scalar *pH_rtDy = g.rtDy.hostPtr();

	for (NC_uint row = 0; row < nmp; row++)
	{
		pH_rtDy[row] = r_t(row);
	}

	g.rtDy.copyToDevice();

	assert(pNC->calcSearchDir(g.kkt, g.rtDy) == 0); // TODO: error

	g.rtDy.copyToHost();

	for (NC_uint row = 0; row < nmp; row++)
	{
		Dy(row) = pH_rtDy[row];
	}
}

IPM_Scalar gpuwrap_calcMaxScaleBTLS(NumCalc_GPU *pNC, IPM_Vector_IN Dlmd)
{
	// TODO: subMat, subVec
	const NC_uint m = NC_uint(Dlmd.rows());

	g.Dlmd.resize(m);

	NC_Scalar *pH_Dlmd = g.Dlmd.hostPtr();

	for (NC_uint row = 0; row < m; row++) {
		pH_Dlmd[row] = Dlmd(row);
	}

	g.Dlmd.copyToDevice();

	IPM_Scalar s_max;
	assert(pNC->calcMaxScaleBTLS(g.lmd, g.Dlmd, &s_max) == 0); // TODO: error

	return s_max;
}

