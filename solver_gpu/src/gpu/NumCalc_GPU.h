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

#ifndef _NUM_CALC_GPU_H_
#define _NUM_CALC_GPU_H_

#include <cuda_runtime.h>
#include <cusolverDn.h>

typedef double NC_SCALAR;
typedef unsigned int NC_uint;

/********************************************************************/

class NCBuf_GPU
{
public:
	explicit NCBuf_GPU() :
		m_nSize(0), m_pBuf(NULL), m_pBufHost(NULL)
	{}
	virtual ~NCBuf_GPU()
	{
		free();
	}
private:
	// uncopyable
	NCBuf_GPU(const NCBuf_GPU&);
	NCBuf_GPU& operator=(const NCBuf_GPU&);

public:
	void realloc(NC_uint nSize)
	{
		if (m_nSize < nSize)
		{
			free();
			m_nSize = nSize;
			if (cudaMalloc(&m_pBuf, nSize)) {
				m_nSize = 0;
				m_pBuf = NULL;
			}
		}
	}

	template < class TYPE = void* > TYPE ptr(void)
	{
		return reinterpret_cast<TYPE>(m_pBuf);
	}

	template < class TYPE = void* > TYPE hostPtr(void)
	{
		if (!m_pBufHost) {
			m_pBufHost = new char[m_nSize];
		}

		return reinterpret_cast<TYPE>(m_pBufHost);
	}

	int copyToHost(void)
	{
		if (m_pBuf) {
			void *pHost = hostPtr();
			if (!cudaMemcpy(pHost, m_pBuf, m_nSize, cudaMemcpyDeviceToHost)) {
				return 0;
			}
		}
		return 1;
	}

	int copyToDevice(void)
	{
		if (m_pBuf && m_pBufHost) {
			if (!cudaMemcpy(m_pBuf, m_pBufHost, m_nSize, cudaMemcpyHostToDevice)) {
				return 0;
			}
		}
		return 1;
	}

	int setZero(void)
	{
		if (m_pBuf) {
			if (!cudaMemset(m_pBuf, 0, m_nSize)) {
				return 0;
			}
		}
		return 1;
	}

protected:
	void free()
	{
		if (m_pBuf) {
			cudaFree(m_pBuf);
			m_pBuf = NULL;
		}
		if (m_pBufHost) {
			delete m_pBufHost;
			m_pBufHost = NULL;
		}
		m_nSize = 0;
	}

	NC_uint  m_nSize;
	void    *m_pBuf;
	char    *m_pBufHost;
};

/********************************************************************/

class NCMat_GPU : public NCBuf_GPU
{
public:
	explicit NCMat_GPU(NC_uint nRows, NC_uint sRows, NC_uint nCols, NC_uint sCols) :
		m_nRows(nRows), m_nRowsPitch(sRows), m_nCols(nCols), m_nColsPitch(sCols)
	{
		realloc(sizeof(NC_SCALAR) * sRows * sCols);
	}
	virtual ~NCMat_GPU() {}
private:
	// uncopyable
	NCMat_GPU(const NCMat_GPU&);
	NCMat_GPU& operator=(const NCMat_GPU&);

public:
	NC_uint nRows(void) { return m_nRows; }
	NC_uint nRowsPitch(void) { return m_nRowsPitch; }
	NC_uint nCols(void) { return m_nCols; }
	NC_uint nColsPitch(void) { return m_nColsPitch; }

protected:
	NC_uint m_nRows; // number of rows
	NC_uint m_nRowsPitch;
	NC_uint m_nCols; // number of columns
	NC_uint m_nColsPitch;
};

/********************************************************************/

class NCVec_GPU : public NCMat_GPU
{
public:
	explicit NCVec_GPU(NC_uint nRows, NC_uint sRows) :
		NCMat_GPU(nRows, sRows, 1, 1)
	{}

	virtual ~NCVec_GPU() {}
private:
	// uncopyable
	NCVec_GPU(const NCVec_GPU&);
	NCVec_GPU& operator=(const NCVec_GPU&);
};

/********************************************************************/

class NumCalc_GPU
{
public:
	explicit NumCalc_GPU();
	virtual ~NumCalc_GPU();

	static void resetDevice(void);

	int calcSearchDirection(NCMat_GPU &kkt, NCVec_GPU &rtDy); // IO, IO

private:
	int m_fatalErr;

	cudaStream_t       m_cuStream;
	cusolverDnHandle_t m_cuSolverDn;
	cublasHandle_t     m_cuBlas;

	NCBuf_GPU m_devInfo;
	NCBuf_GPU m_solverBuf;
	NCBuf_GPU m_solverTau;
};

#endif // end of ifndef _NUM_CALC_GPU_H_
