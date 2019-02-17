

#include "QP.h"
#include "QCQP.h"
#include "SOCP.h"

#include <iostream>
#include <fstream>
using namespace std;


static void testQP(void)
{
	cout << endl << __FUNCTION__ << endl;

	const IPM_uint n = 2; // x0, x1
	const IPM_uint m = 1;
	const IPM_uint p = 0;
	//
	IPM_Vector x(n);
	IPM_Matrix P(n, n);
	IPM_Vector q(n);
	IPM_Single r;
	IPM_Matrix G(m, n);
	IPM_Vector h(m, 1);
	IPM_Matrix A(p, n);
	IPM_Vector b(p, 1);
	x.setZero();
	P.setZero();
	q.setZero();
	r.setZero();
	A.setZero();
	b.setZero();
	// (1/2)(x - a)^2 + const
	P(0, 0) = 1.0;
	P(1, 1) = 1.0;
	q(0, 0) = -(-1.0); // a0
	q(1, 0) = -(-2.0); // a1
	// 1 - x0/b0 - x1/b1 <= 0
	G(0, 0) = -1.0 / (2.0); // b0
	G(0, 1) = -1.0 / (3.0); // b1
	h(0) = -1.0;


	QP instance;

	ofstream ofs("logQP.txt");
	instance.setLog(&ofs);

	IPM_Error err = instance.solve(x, P, q, r, G, h, A, b);

	if (err)
	{
		cout << "!!!!! " << err << endl;
	}
	else
	{
		cout << x << endl;
		cout << "converged: " << instance.isConverged() << endl;
	}

	ofs.close();
}

static void testQCQP(void)
{
	cout << endl << __FUNCTION__ << endl;

	const IPM_uint n = 2; // x0, x1
	const IPM_uint m = 1;
	const IPM_uint p = 0;
	//
	IPM_Vector x(n);
	IPM_Matrix a_P[m + 1];
	IPM_Vector a_q[m + 1];
	IPM_Single a_r[m + 1];
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
	// (1/2)(x - a)^2 + const
	a_P[0](0, 0) = 1.0;
	a_P[0](1, 1) = 1.0;
	a_q[0](0, 0) = -(5.0); // a0
	a_q[0](1, 0) = -(4.0); // a1
	// 1 - x0/b0 - x1/b1 <= 0
	a_q[1](0, 0) = -1.0 / (2.0); // b0
	a_q[1](1, 0) = -1.0 / (3.0); // b1
	a_r[1](0, 0) = 1.0;


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
		cout << x << endl;
		cout << "converged: " << instance.isConverged() << endl;
	}

	ofs.close();
}

static void testSOCP(void)
{
	cout << endl << __FUNCTION__ << endl;

	const IPM_uint n = 2; // x0, x1
	const IPM_uint m = 1;
	const IPM_uint p = 0;
	//
	IPM_Vector x(n);
	IPM_Vector f(n);
	IPM_Matrix a_G[m];
	IPM_Vector a_h[m];
	IPM_Vector a_c[m];
	IPM_Single a_d[m];
	IPM_Matrix A(p, n);
	IPM_Vector b(p, 1);
	x.setZero();
	f.setOnes();
	for (IPM_uint i = 0; i < m; i++)
	{
		IPM_uint ni = 2;
		a_G[i] = IPM_Matrix(ni, n);
		a_h[i] = IPM_Vector(ni);
		a_c[i] = IPM_Vector(n);
		a_d[i] = IPM_Single();
		a_G[i].setZero();
		a_h[i].setZero();
		a_c[i].setZero();
		a_d[i].setZero();
	}
	A.setZero();
	b.setZero();
	//
	a_G[0](0, 0) = 1.0;
	a_G[0](1, 1) = 1.0;
	a_c[0](0, 0) = 0.0;
	a_c[0](1, 0) = 0.0;
	a_d[0](0, 0) = 1.41421356;


	SOCP instance;

	ofstream ofs("logSOCP.txt");
	instance.setLog(&ofs);

	IPM_Error err = instance.solve(x, f, a_G, a_h, a_c, a_d, m, A, b);

	if (err)
	{
		cout << "!!!!! " << err << endl;
	}
	else
	{
		cout << x << endl;
		cout << "converged: " << instance.isConverged() << endl;
	}

	ofs.close();
}

static void testInfeasible(void)
{
	cout << endl << __FUNCTION__ << endl;

	const IPM_uint n = 1;
	const IPM_uint m = 2;
	const IPM_uint p = 0;
	//
	IPM_Vector x(n);
	IPM_Matrix a_P[m + 1];
	IPM_Vector a_q[m + 1];
	IPM_Single a_r[m + 1];
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
	// (1/2)(x - a)^2 + const
	//a_P[0](0, 0) = 1.0;
	//a_q[0](0, 0) = -(5.0); // a
	// x <= b, x >= c
	a_q[1](0, 0) = 1.0;
	a_r[1](0, 0) = -(5.0); // b
	a_q[2](0, 0) = -1.0;
	a_r[2](0, 0) = (10.0); // c


	QCQP instance;

	ofstream ofs("logInfeasible.txt");
	instance.setLog(&ofs);

	IPM_Error err = instance.solve(x, a_P, a_q, a_r, m, A, b);

	if (err)
	{
		cout << "!!!!! " << err << endl;
	}
	else
	{
		cout << x << endl;
		cout << "converged: " << instance.isConverged() << endl;
	}

	ofs.close();
}

int main(int argc, char **argv)
{
	testQP();
	testQCQP();
	testSOCP();
	testInfeasible();

	cout << endl << "hit enter to exit" << endl;
	getchar();
}