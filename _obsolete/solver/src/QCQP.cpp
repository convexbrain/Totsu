

#include "QCQP.h"


QCQP::QCQP()
{
	m_slack = m_margin;

	m_converged = true;
}

QCQP::~QCQP()
{
	// do nothing
}

IPM_Error QCQP::initialPoint(IPM_Vector_IO x)
{
	const IPM_uint _n = x.size() - 1;

	x.head(_n) = *m_p_x;

	// a slack variable
	IPM_Scalar s = 0;
	IPM_Single tmp;
	for (IPM_uint i = 1; i <= m_m; i++)
	{
		tmp = (*m_p_x).transpose() * m_p_P[i] * (*m_p_x) / IPM_Scalar(2.0)
			  + m_p_q[i].transpose() * (*m_p_x)
			  + m_p_r[i];
		s = max(s, tmp(0, 0));
	}
	IPM_Scalar slack = m_slack;
	IPM_Scalar s_slack;
	do {
		s_slack = s + slack;
		slack *= IPM_Scalar(2.0);
	} while (!(s_slack > s));

	x(_n) = s_slack;

	return NULL;
}

IPM_Error QCQP::finalPoint(IPM_Vector_IN x, IPM_Vector_IN lmd, IPM_Vector_IN nu, bool converged)
{
	const IPM_uint _n = x.size() - 1;

	*m_p_x = x.head(_n);

	m_converged = converged;

	return NULL;
}

IPM_Error QCQP::objective(IPM_Vector_IN x, IPM_Single_IO f_o)
{
	const IPM_uint _n = x.size() - 1;

	f_o = x.head(_n).transpose() * m_p_P[0] * x.head(_n) / IPM_Scalar(2.0)
		  + m_p_q[0].transpose() * x.head(_n)
		  + m_p_r[0];

	return NULL;
}

IPM_Error QCQP::Dobjective(IPM_Vector_IN x, IPM_Vector_IO Df_o)
{
	const IPM_uint _n = x.size() - 1;

	Df_o.head(_n) = m_p_P[0] * x.head(_n) + m_p_q[0];

	// for a slack variable
	Df_o(_n, 0) = 0;

	return NULL;
}

IPM_Error QCQP::DDobjective(IPM_Vector_IN x, IPM_Matrix_IO DDf_o)
{
	const IPM_uint _n = x.size() - 1;

	DDf_o.topLeftCorner(_n, _n) = m_p_P[0];

	// for a slack variable
	DDf_o.bottomRows(1).setZero();
	DDf_o.rightCols(1).setZero();

	return NULL;
}

IPM_Error QCQP::inequality(IPM_Vector_IN x, IPM_Vector_IO f_i)
{
	const IPM_uint _n = x.size() - 1;

	IPM_Single tmp;
	for (int r = 0; r < f_i.size(); r++)
	{
		IPM_uint i = r + 1;
		tmp = x.head(_n).transpose() * m_p_P[i] * x.head(_n) / IPM_Scalar(2.0)
			  + m_p_q[i].transpose() * x.head(_n)
			  + m_p_r[i];
		f_i(r, 0) = tmp(0, 0) - x(_n, 0); // minus a slack variable

	}

	return NULL;
}

IPM_Error QCQP::Dinequality(IPM_Vector_IN x, IPM_Matrix_IO Df_i)
{
	const IPM_uint _n = x.size() - 1;

	for (int r = 0; r < Df_i.rows(); r++)
	{
		IPM_uint i = r + 1;
		Df_i.block(r, 0, 1, _n) = (m_p_P[i] * x.head(_n) + m_p_q[i]).transpose();

		// for a slack variable
		Df_i(r, _n) = -1.0;
	}

	return NULL;
}

IPM_Error QCQP::DDinequality(IPM_Vector_IN x, IPM_Matrix_IO DDf_i, const IPM_uint of_i)
{
	const IPM_uint _n = x.size() - 1;

	DDf_i.topLeftCorner(_n, _n) = m_p_P[of_i + 1];

	// for a slack variable
	DDf_i.bottomRows(1).setZero();
	DDf_i.rightCols(1).setZero();

	return NULL;
}

IPM_Error QCQP::equality(IPM_Matrix_IO A, IPM_Vector_IO b)
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

IPM_Error QCQP::solve(IPM_Vector& x,
	const IPM_Matrix P[], const IPM_Vector q[], const IPM_Single r[], const IPM_uint m,
	const IPM_Matrix& A, const IPM_Vector& b)
{
	const IPM_uint n = x.size();
	const IPM_uint p = A.rows();

	// ----- parameter check

	if (x.size() == 0) return IPM_ERR_STR;

	if (!P) return IPM_ERR_STR;
	if (!q) return IPM_ERR_STR;
	if (!r) return IPM_ERR_STR;

	// m = 0 means NO inequality constraints
	for (IPM_uint i = 0; i <= m; i++)
	{
		if (P[i].rows() != P[i].cols()) return IPM_ERR_STR;
		if ((IPM_uint)P[i].rows() != n) return IPM_ERR_STR;
		if ((IPM_uint)q[i].size() != n) return IPM_ERR_STR;
	}

	// p = 0 means NO equality constraints
	if (p > 0)
	{
		if ((IPM_uint)A.cols() != n) return IPM_ERR_STR;

		if ((IPM_uint)b.size() != p) return IPM_ERR_STR;
	}

	// ----- set member variables

	m_p_x = &x;
	m_m = m;
	m_p_P = P;
	m_p_q = q;
	m_p_r = r;
	m_p_A = &A;
	m_p_b = &b;

	// ----- start to solve

	// '+ 1' is for a slack variable
	IPM_Error err = start(n + 1, m, p + 1);
	if (err) return err;

	return NULL;
}
