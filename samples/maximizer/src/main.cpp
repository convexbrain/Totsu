

#include "QCQP.h"

#include <iostream>
#include <fstream>
using namespace std;


int main(int argc, char **argv)
{
	const int ch = 2;
	const int sz = sizeof(short);
	const int len_max = 1000000;
	short *pPcmIn = new short[len_max * ch];

	FILE *fpIn = fopen("../../../../data/maximizer/in.raw", "rb");
	FILE *fpOut = fopen("out.raw", "wb");

	int len_read = fread(pPcmIn, ch * sz, len_max, fpIn);

	//

	const IPM_Scalar scl = 4.0;
	const IPM_uint len_opt = 16;
	const IPM_uint len_out = 16;

	//

	const IPM_uint n = len_opt * ch + ch;
	const IPM_uint m = len_opt * ch * 2;
	const IPM_uint p = ch;
	IPM_Vector x(n);
	IPM_Matrix a_P[m + 1];
	IPM_Vector a_q[m + 1];
	IPM_Single a_r[m + 1];
	IPM_Matrix A(p, n);
	IPM_Vector b(p);

	QCQP instance;

	ofstream ofs("log.txt");
	//instance.setLog(&ofs);

	for (int k = 0; k <= m; k++)
	{
		a_P[k] = IPM_Matrix(n, n);
		a_q[k] = IPM_Vector(n);
		a_r[k] = IPM_Single();
	}

	IPM_Matrix D = IPM_Matrix::Identity(n - ch, n) * (-1.0);
	for (int k = 0; k + ch < n; k++)
	{
		D(k, k + ch) = 1.0;
	}
	a_P[0] = D.transpose() * D;
	a_r[0].setZero();

	for (int k = 1; k <= m; k++)
	{
		a_P[k].setZero();
		a_q[k].setZero();
		a_r[k].setOnes();
		a_r[k] *= -1.0;
	}
	for (int k = 0; k < len_opt * ch; k++)
	{
		a_q[1 + k](k) = 1.0;
		a_q[1 + k + len_opt * ch](k) = -1.0;
	}

	A.setIdentity();

	//

	IPM_Vector xIn(n);

	short *ptr = pPcmIn;
	short *ptrEnd = &pPcmIn[len_read * ch];
	short out[len_out * ch];
	IPM_Scalar prevPcmIn[ch], prevPcmOut[ch];
	for (int k = 0; k < ch; k++)
	{
		prevPcmIn[k] = prevPcmOut[k] = 0;
	}

	ptr += 39000 * ch;
	int num = 0;
	while (ptr < ptrEnd - n)
	{
		cout << num << endl;

		for (int k = 0; k < ch; k++)
		{
			xIn(k) = prevPcmIn[k];
			x(k) = prevPcmOut[k];
			b(k) = prevPcmOut[k];
		}
		for (int k = ch; k < n; k++)
		{
			xIn(k) = ptr[k - ch] / 32768.0 * scl;
			x(k) = ptr[k - ch] / 32768.0 * scl;
		}

		a_q[0] = (-1.0) * a_P[0] * xIn;
		IPM_Error err = instance.solve(x, a_P, a_q, a_r, m, A, b);
		if (err)
		{
			cout << "!!!!! " << err << endl;
			break;
		}

		for (int k = 0; k < len_out * ch; k++)
		{
			int o = (x(ch + k) * 32768.0 + 0.5);
			if (o > 32767) o = 32767;
			if (o < -32768) o = -32768;
			out[k] = o;
		}
		for (int k = 0; k < ch; k++)
		{
			prevPcmIn[k] = xIn(len_out * ch + k);
			prevPcmOut[k] = x(len_out * ch + k);
		}

		fwrite(out, ch * sz, len_out, fpOut);
		//fflush(fpOut);
		//if (num > 2) break;

		ptr += len_out * ch;
		num += len_out;
	}

	ofs.close();
	fclose(fpIn);
	fclose(fpOut);
	delete[] pPcmIn;

	return 0;
}
