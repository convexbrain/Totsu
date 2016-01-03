

#include "SOCP.h"


SOCP::SOCP()
{
	m_eps_bd = m_eps;
	m_slack = m_margin;

	m_converged = true;
}

SOCP::~SOCP()
{
	// do nothing
}

IPM_Error SOCP::initialPoint(IPM_Vector_IO x)
{
	const IPM_uint _n = x.size() - m_m;

	x.head(_n) = *m_p_x;

	// slack variables
	for (IPM_uint i = 0; i < m_m; i++)
	{
		IPM_Vector tmp = m_p_G[i] * (*m_p_x) + m_p_h[i];
		IPM_Scalar s = tmp.norm() + m_eps_bd;

		IPM_Scalar slack = m_slack;
		IPM_Scalar s_slack;
		do {
			s_slack = s + slack;
			slack *= IPM_Scalar(2.0);
		} while (!(s_slack > s));
		x(_n + i) = s_slack;
	}

	return NULL;
}

IPM_Error SOCP::finalPoint(IPM_Vector_IN x, IPM_Vector_IN lmd, IPM_Vector_IN nu, bool converged)
{
	const IPM_uint _n = x.size() - m_m;

	*m_p_x = x.head(_n);

	m_converged = converged;

	return NULL;
}

IPM_Error SOCP::objective(IPM_Vector_IN x, IPM_Single_IO f_o)
{
	const IPM_uint _n = x.size() - m_m;

	f_o = m_p_f->transpose() * x.head(_n);

	return NULL;
}

IPM_Error SOCP::Dobjective(IPM_Vector_IN x, IPM_Vector_IO Df_o)
{
	const IPM_uint _n = x.size() - m_m;

	Df_o.head(_n) = *m_p_f;

	// for slack variables
	Df_o.tail(m_m).setZero();

	return NULL;
}

IPM_Error SOCP::DDobjective(IPM_Vector_IN x, IPM_Matrix_IO DDf_o)
{
	DDf_o.setZero();

	return NULL;
}

IPM_Error SOCP::inequality(IPM_Vector_IN x, IPM_Vector_IO f_i)
{
	const IPM_uint _n = x.size() - m_m;

	for (IPM_uint r = 0; r < m_m; r++)
	{
		IPM_Scalar inv_s;
		if (abs(x(_n + r)) > m_eps)
		{
			inv_s = IPM_Scalar(1.0 / x(_n + r));
		}
		else
		{
			inv_s = IPM_Scalar(1.0 / m_eps);
			logWarn("guard from div by zero");
		}

		IPM_Vector tmp = m_p_G[r] * x.head(_n) + m_p_h[r];
		f_i(r) = tmp.squaredNorm() * inv_s - x(_n + r);

		// for slack variables
		f_i(r + m_m) = m_eps_bd - x(_n + r);
	}

	return NULL;
}

IPM_Error SOCP::Dinequality(IPM_Vector_IN x, IPM_Matrix_IO Df_i)
{
	const IPM_uint _n = x.size() - m_m;

	Df_i.setZero();

	for (IPM_uint r = 0; r < m_m; r++)
	{
		IPM_Scalar inv_s;
		if (abs(x(_n + r)) > m_eps)
		{
			inv_s = IPM_Scalar(1.0 / x(_n + r));
		}
		else
		{
			inv_s = IPM_Scalar(1.0 / m_eps);
			logWarn("guard from div by zero");
		}

		IPM_Vector tmp1 = m_p_G[r] * x.head(_n) + m_p_h[r];
		IPM_Vector tmp2 = 2 * inv_s * m_p_G[r].transpose() * tmp1;
		Df_i.block(r, 0, 1, _n) = tmp2.transpose();

		// for slack variables
		Df_i(r, _n + r) = -inv_s * inv_s * tmp1.squaredNorm() - 1;

		// for slack variables
		Df_i(r + m_m, _n + r) = -1;
	}

	return NULL;
}

IPM_Error SOCP::DDinequality(IPM_Vector_IN x, IPM_Matrix_IO DDf_i, const IPM_uint of_i)
{
	const IPM_uint _n = x.size() - m_m;

	if (of_i < m_m)
	{
		DDf_i.setZero();

		IPM_Scalar inv_s;
		if (abs(x(_n + of_i)) > m_eps)
		{
			inv_s = IPM_Scalar(1.0 / x(_n + of_i));
		}
		else
		{
			inv_s = IPM_Scalar(1.0 / m_eps);
			logWarn("guard from div by zero");
		}

		DDf_i.topLeftCorner(_n, _n) = 2 * inv_s * m_p_G[of_i].transpose() * m_p_G[of_i];

		IPM_Vector tmp1 = m_p_G[of_i] * x.head(_n) + m_p_h[of_i];
		IPM_Vector tmp2 = -2 * inv_s * inv_s * m_p_G[of_i].transpose() * tmp1;

		// for slack variables
		DDf_i.block(0, _n + of_i, _n, 1) = tmp2;

		// for slack variables
		DDf_i.block(_n + of_i, 0, 1, _n) = tmp2.transpose();

		// for slack variables
		DDf_i(_n + of_i, _n + of_i) = 2 * inv_s * inv_s * inv_s * tmp1.squaredNorm();
	}
	else
	{
		// for slack variables
		DDf_i.setZero();
	}

	return NULL;
}

IPM_Error SOCP::equality(IPM_Matrix_IO A, IPM_Vector_IO b)
{
	const IPM_uint _n = A.cols() - m_m;
	const IPM_uint _p = A.rows() - m_m;

	A.setZero();

	A.topLeftCorner(_p, _n) = *m_p_A;
	b.head(_p) = *m_p_b;

	for(IPM_uint r = 0; r < m_m; r++)
	{
		A.block(_p + r, 0, 1, _n) = m_p_c[r].transpose();
		A(_p + r, _n + r) = -1;
		b(_p + r) = -m_p_d[r](0, 0);
	}

	return NULL;
}

IPM_Error SOCP::solve(IPM_Vector& x,
	const IPM_Vector& f,
	const IPM_Matrix G[], const IPM_Vector h[], const IPM_Vector c[], const IPM_Single d[], const IPM_uint m,
	const IPM_Matrix& A, const IPM_Vector& b)
{
	const IPM_uint n = x.size();
	const IPM_uint p = A.rows();

	// ----- parameter check

	if (x.size() == 0) return IPM_ERR_STR;

	if (!G) return IPM_ERR_STR;
	if (!h) return IPM_ERR_STR;
	if (!c) return IPM_ERR_STR;
	if (!d) return IPM_ERR_STR;

	// m = 0 means NO inequality constraints
	for (IPM_uint i = 0; i < m; i++)
	{
		if ((IPM_uint)G[i].cols() != n) return IPM_ERR_STR;
		if (G[i].rows() != h[i].rows()) return IPM_ERR_STR;
		if ((IPM_uint)c[i].rows() != n) return IPM_ERR_STR;
	}

	// p = 0 means NO equality constraints
	if (p > 0)
	{
		if ((IPM_uint)A.cols() != n) return IPM_ERR_STR;

		if ((IPM_uint)b.size() != p) return IPM_ERR_STR;
	}

	// ----- set member variables

	m_p_x = &x;
	m_p_f = &f;
	m_m = m;
	m_p_G = G;
	m_p_h = h;
	m_p_c = c;
	m_p_d = d;
	m_p_A = &A;
	m_p_b = &b;

	// ----- start to solve

	// '+ m' is for a slack variable
	IPM_Error err = start(n + m, m + m, p + m);
	if (err) return err;

	return NULL;
}
