
#include "QCQP.h"

#include <iostream>
#include <fstream>
#include <vector>
using namespace std;


static void testSVM(void)
{
	FILE *fp = fopen("../../../../data/svm/rdat.csv", "rt");

	int dim, num;
	fscanf(fp, "%d,%d,", &dim, &num);

	IPM_Matrix XY(num, dim + 2);
	{
		for (int n = 0; n < num; n++)
		{
			for (int d = 0; d < dim; d++)
			{
				double x;
				fscanf(fp, "%lf,", &x);
				XY(n, d) = x;
			}
			int y;
			fscanf(fp, "%d", &y);
			XY(n, dim) = y;
		}
		//cout << XY;
	}

	fclose(fp);

	//-----

	const IPM_uint n = num;
	const IPM_uint m = num;
	const IPM_uint p = 1;
	IPM_Vector x(n);
	IPM_Matrix *a_P = new IPM_Matrix[m + 1];
	IPM_Vector *a_q = new IPM_Vector[m + 1];
	IPM_Single *a_r = new IPM_Single[m + 1];
	IPM_Matrix A(p, n);
	IPM_Vector b(p, 1);
	x.setZero();
	for (IPM_uint i = 0; i <= m; i++)
	{
		a_P[i] = IPM_Matrix(n, n);
		a_q[i] = IPM_Vector(n);
		a_r[i] = IPM_Single();
		a_P[i].setZero();
		a_q[i].setZero();
		a_r[i].setZero();
	}
	A.setZero();
	b.setZero();

	//-----

	for (IPM_uint r = 0; r < n; r++)
	{
		for (IPM_uint c = 0; c < n; c++)
		{
			IPM_Scalar squaredSigma = 1.0 / 8.0;
			IPM_Scalar squaredNorm = (XY.block(r, 0, 1, dim) - XY.block(c, 0, 1, dim)).squaredNorm();
			IPM_Scalar kernel = exp(-squaredNorm / squaredSigma);
			a_P[0](r, c) = XY(r, dim) * XY(c, dim) * kernel;
		}
	}
	for (IPM_uint r = 0; r < n; r++)
	{
		a_q[0](r) = -1.0;
	}
	for (IPM_uint i = 1; i <= m; i++)
	{
		a_q[i](i - 1) = -1.0;
	}
	for (IPM_uint c = 0; c < n; c++)
	{
		A(0, c) = XY(c, dim);
	}

	//-----

	QCQP instance;

	ofstream ofs("logQCQP.txt");
	instance.setLog(&ofs);

	IPM_Error err = instance.solve(x, a_P, a_q, a_r, m, A, b);

	if (err)
	{
		cout << "!!!!! " << err << endl;
	}
	else
	{
		cout << "converged: " << instance.isConverged() << endl;

		for (IPM_uint i = 0; i < n; i++)
		{
			if (x(i) > 1.0 / 65536.0)
			{
				cout << "[" << x(i);
				for (int d = 0; d < dim + 1; d++)
				{
					cout << "," << XY(i, d);
				}
				cout << "]," << endl;
			}
			else
			{
				x(i) = 0;
			}
		}
		XY.col(dim + 1) = x;

		IPM_Scalar wx_p_min = INFINITY;
		IPM_Scalar wx_n_max = -INFINITY;
		for (IPM_uint r = 0; r < n; r++)
		{
			if (XY(r, dim + 1) > 0)
			{
				IPM_Scalar wx = 0;
				for (IPM_uint i = 0; i < n; i++)
				{
					if (XY(i, dim + 1) > 0)
					{
						IPM_Scalar squaredSigma = 1.0 / 8.0;
						IPM_Scalar squaredNorm = (XY.block(r, 0, 1, dim) - XY.block(i, 0, 1, dim)).squaredNorm();
						IPM_Scalar kernel = exp(-squaredNorm / squaredSigma);
						wx += XY(i, dim + 1) * XY(i, dim) * kernel;
					}
				}
				//cout << XY(r, dim) << " " << wx << endl;

				if (XY(r, dim) > 0) wx_p_min = min(wx_p_min, wx);
				else wx_n_max = max(wx_n_max, wx);
			}
		}
		IPM_Scalar bias = -0.5 * (wx_p_min + wx_n_max);
		cout << bias << endl;
	}

	ofs.close();

	//-----

	delete[] a_P;
	delete[] a_q;
	delete[] a_r;
}

int main(int argc, char **argv)
{
	testSVM();

	cout << endl << "hit enter to exit" << endl;
	getchar();
}
