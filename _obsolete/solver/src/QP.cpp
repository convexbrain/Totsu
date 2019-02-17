

#include "QP.h"


QP::QP()
{
	m_slack = m_margin;

	m_converged = true;
}

QP::~QP()
{
	// do nothing
}

IPM_Error QP::initialPoint(IPM_Vector_IO x)
{
	const IPM_uint _n = x.size() - 1;

	x.head(_n) = *m_p_x;

	// a slack variable
	IPM_Vector tmp = (*m_p_G) * (*m_p_x) - (*m_p_h);
	IPM_Scalar s = tmp.maxCoeff();

	IPM_Scalar slack = m_slack;
	IPM_Scalar s_slack;
	do {
		s_slack = s + slack;
		slack *= IPM_Scalar(2.0);
	} while (!(s_slack > s));

	x(_n) = s_slack;

	return NULL;
}

IPM_Error QP::finalPoint(IPM_Vector_IN x, IPM_Vector_IN lmd, IPM_Vector_IN nu, bool converged)
{
	const IPM_uint _n = x.size() - 1;

	*m_p_x = x.head(_n);

	m_converged = converged;

	return NULL;
}

IPM_Error QP::objective(IPM_Vector_IN x, IPM_Single_IO f_o)
{
	const IPM_uint _n = x.size() - 1;

	f_o = x.head(_n).transpose() * (*m_p_P) * x.head(_n) / IPM_Scalar(2.0)
		  + (*m_p_q).transpose() * x.head(_n)
		  + (*m_p_r);

	return NULL;
}

IPM_Error QP::Dobjective(IPM_Vector_IN x, IPM_Vector_IO Df_o)
{
	const IPM_uint _n = x.size() - 1;

	Df_o.head(_n) = (*m_p_P) * x.head(_n) + (*m_p_q);

	// for a slack variable
	Df_o(_n, 0) = 0;

	return NULL;
}

IPM_Error QP::DDobjective(IPM_Vector_IN x, IPM_Matrix_IO DDf_o)
{
	const IPM_uint _n = x.size() - 1;

	DDf_o.topLeftCorner(_n, _n) = *m_p_P;

	// for a slack variable
	DDf_o.bottomRows(1).setZero();
	DDf_o.rightCols(1).setZero();

	return NULL;
}

IPM_Error QP::inequality(IPM_Vector_IN x, IPM_Vector_IO f_i)
{
	const IPM_uint _n = x.size() - 1;
	const IPM_uint _m = f_i.size();

	f_i = (*m_p_G) * x.head(_n) - (*m_p_h)
		  - x(_n) * IPM_Vector::Ones(_m); // minus a slack variable

	return NULL;
}

IPM_Error QP::Dinequality(IPM_Vector_IN x, IPM_Matrix_IO Df_i)
{
	const IPM_uint _n = x.size() - 1;
	const IPM_uint _m = Df_i.rows();

	Df_i.leftCols(_n) = *m_p_G;

	// for a slack variable
	Df_i.rightCols(1) = -IPM_Vector::Ones(_m);

	return NULL;
}

IPM_Error QP::DDinequality(IPM_Vector_IN x, IPM_Matrix_IO DDf_i, const IPM_uint of_i)
{
	DDf_i.setZero();

	return NULL;
}

IPM_Error QP::equality(IPM_Matrix_IO A, IPM_Vector_IO b)
{
	const IPM_uint _n = A.cols() - 1;
	const IPM_uint _p = A.rows() - 1;

	A.topLeftCorner(_p, _n) = *m_p_A;
	b.head(_p) = *m_p_b;

	// for a slack variable
	A.bottomRows(1).setZero();
	A.rightCols(1).setZero();
	A(_p, _n) = 1;
	b(_p, 0) = 0;

	return NULL;
}

IPM_Error QP::solve(IPM_Vector& x,
	const IPM_Matrix& P, const IPM_Vector& q, const IPM_Single& r,
	const IPM_Matrix& G, const IPM_Vector& h,
	const IPM_Matrix& A, const IPM_Vector& b)
{
	const IPM_uint n = x.size();
	const IPM_uint m = G.rows();
	const IPM_uint p = A.rows();

	// ----- parameter check

	if (x.size() == 0) return IPM_ERR_STR;

	if (P.rows() != P.cols()) return IPM_ERR_STR;
	if ((IPM_uint)P.rows() != n) return IPM_ERR_STR;
	if ((IPM_uint)q.size() != n) return IPM_ERR_STR;

	// m = 0 means NO inequality constraints
	if (m > 0)
	{
		if ((IPM_uint)G.cols() != n) return IPM_ERR_STR;

		if ((IPM_uint)h.size() != m) return IPM_ERR_STR;
	}

	// p = 0 means NO equality constraints
	if (p > 0)
	{
		if ((IPM_uint)A.cols() != n) return IPM_ERR_STR;

		if ((IPM_uint)b.size() != p) return IPM_ERR_STR;
	}

	// ----- set member variables

	m_p_x = &x;
	m_p_P = &P;
	m_p_q = &q;
	m_p_r = &r;
	m_p_G = &G;
	m_p_h = &h;
	m_p_A = &A;
	m_p_b = &b;

	// ----- start to solve

	// '+ 1' is for a slack variable
	IPM_Error err = start(n + 1, m, p + 1);
	if (err) return err;

	return NULL;
}
