

#include "PrimalDualIPM.h"

#define DEV_GPU // TODO
#include "gpu/declarations.h"
#include "gpu/NumCalc_GPU.h"


#ifndef IPM_DISABLE_LOG
// iostream and std is just for logging
#include <iostream>
#define IPM_LOG_EN(...) {using namespace std; __VA_ARGS__}
#else
#define IPM_LOG_EN(...) /**/
#endif


PrimalDualIPM::PrimalDualIPM()
{
	IPM_LOG_EN({
		m_pOuts = NULL;
	});

	m_margin = IPM_Scalar(1.0);

	m_loop = 256;
	m_bloop = 256;
	m_eps_feas = IPM_Scalar(sqrt(IPM_SCALAR_EPS)); // TODO
	m_eps = IPM_Scalar(sqrt(IPM_SCALAR_EPS)); // TODO

	m_mu = IPM_Scalar(10.0);
	m_alpha = IPM_Scalar(0.1);
	m_beta = IPM_Scalar(0.8);

	m_s_coef = IPM_Scalar(0.99);

	m_pNumCalc = new NumCalc_GPU;
}

PrimalDualIPM::~PrimalDualIPM()
{
	delete m_pNumCalc;
	NumCalc_GPU::resetDevice();
}

void PrimalDualIPM::logWarn(const char *str)
{
	IPM_LOG_EN({
		if (!m_pOuts) return;

		*m_pOuts << "----- WARNING: " << str << endl;
	});
}

void PrimalDualIPM::logVector(const char *str, IPM_Vector_IN v)
{
	IPM_LOG_EN({
		if (!m_pOuts) return;

		if (v.cols() == 1)
		{
			*m_pOuts << "----- " << str << "^T :" << endl;
			*m_pOuts << v.transpose() << endl;
		}
		else
		{
			*m_pOuts << "----- " << str << " :" << endl;
			*m_pOuts << v << endl;
		}
	});
}

void PrimalDualIPM::logMatrix(const char *str, IPM_Matrix_IN m)
{
	IPM_LOG_EN({
		if (!m_pOuts) return;

		*m_pOuts << "----- " << str << " :" << endl;
		*m_pOuts << m << endl;
	});
}

IPM_Error PrimalDualIPM::start(const IPM_uint n, const IPM_uint m, const IPM_uint p)
{
	IPM_Error err;
	bool converged = true;

	// parameter check
	if (n == 0) return IPM_ERR_STR;

	/***** matrix *****/
	// constant across loop
	IPM_Matrix A(p, n);
	IPM_Vector b(p);
	// loop variable
	IPM_Vector y(n + m + p), Dy(n + m + p);
	//IPM_Matrix kkt(n + m + p, n + m + p);
	// temporal in loop
	IPM_Vector Df_o(n), f_i(m), r_t(n + m + p), y_p(n + m + p);
	IPM_Matrix Df_i(m, n), DDf(n, n);

	/***** sub matrix *****/
	IPM_Vector_IO x = y.segment(0, n);
	IPM_Vector_IO lmd = y.segment(n, m);
	IPM_Vector_IO nu = y.segment(n + m, p);
	IPM_Vector_IO r_dual = r_t.segment(0, n);
	IPM_Vector_IO r_cent = r_t.segment(n, m);
	IPM_Vector_IO r_pri = r_t.segment(n + m, p);
	//IPM_Matrix_IO kkt_x_dual = kkt.block(0, 0, n, n);
	//IPM_Matrix_IO kkt_lmd_dual = kkt.block(0, n, n, m);
	//IPM_Matrix_IO kkt_nu_dual = kkt.block(0, n + m, n, p);
	//IPM_Matrix_IO kkt_x_cent = kkt.block(n, 0, m, n);
	//IPM_Matrix_IO kkt_lmd_cent = kkt.block(n, n, m, m);
	//IPM_Matrix_IO kkt_x_pri = kkt.block(n + m, 0, p, n);
	IPM_Vector_IO Dlmd = Dy.segment(n, m);
	IPM_Vector_IO x_p = y_p.segment(0, n);
	IPM_Vector_IO lmd_p = y_p.segment(n, m);
	IPM_Vector_IO nu_p = y_p.segment(n + m, p);

	// initialize
	if ((err = initialPoint(x)) != NULL) return err;
	lmd.setOnes();
	lmd *= m_margin;
	nu.setZero();
	if ((err = equality(A, b)) != NULL) return err;

	// initial Df_o, f_i, Df_i
	if ((err = Dobjective(x, Df_o)) != NULL) return err;
	if ((err = inequality(x, f_i)) != NULL) return err;
	if ((err = Dinequality(x, Df_i)) != NULL) return err;

	// inequality feasibility check
	if (f_i.maxCoeff() >= 0) return IPM_ERR_STR;

	// initial residual - dual and primal
	r_dual = Df_o;
	if (m > 0) r_dual += Df_i.transpose() * lmd;
	if (p > 0) r_dual += A.transpose() * nu;
	if (p > 0) r_pri = A * x - b;


	IPM_uint loop = 0;
	for (; loop < m_loop; loop++)
	{
		IPM_Scalar eta, inv_t;

		IPM_LOG_EN({ if (m_pOuts) *m_pOuts << endl << "===== ===== ===== ===== loop : " << loop << endl; });

		gpuwrap_set_y(y, n, m, p);
		gpuwrap_set_f_i(f_i);

		/***** calc t *****/

		eta = m_eps;
		if (m > 0) eta = gpuwrap_eta(m_pNumCalc);

		// inequality feasibility check
		if (eta < 0) return IPM_ERR_STR;

		inv_t = 0;
		if (m > 0) inv_t = eta / (m_mu * m);

		/***** update residual - central *****/

		if (m > 0) gpuwrap_r_cent(m_pNumCalc, r_cent, inv_t);

		/***** termination criteria *****/

		gpuwrap_set_r_t(r_t, n, m, p);
		IPM_Scalar org_r_t_norm = gpuwrap_r_t_norm(m_pNumCalc); // used in back tracking line search

		IPM_Scalar r_dual_norm = gpuwrap_r_dual_norm(m_pNumCalc);
		IPM_Scalar r_pri_norm = gpuwrap_r_pri_norm(m_pNumCalc);
		IPM_LOG_EN({ if (m_pOuts) *m_pOuts << "|| r_dual || : " << r_dual_norm << endl; });
		IPM_LOG_EN({ if (m_pOuts) *m_pOuts << "|| r_pri  || : " << r_pri_norm << endl; });
		IPM_LOG_EN({ if (m_pOuts) *m_pOuts << "   eta       : " << eta << endl; });
		if ((r_dual_norm <= m_eps_feas) && (r_pri_norm <= m_eps_feas) && (eta <= m_eps))
		{
			IPM_LOG_EN({ if (m_pOuts) *m_pOuts << "termination criteria satisfied" << endl; });
			break;
		}

		/***** calc kkt matrix *****/

		gpuwrap_clearKKT(m_pNumCalc, n, m, p);

		if ((err = DDobjective(x, DDf)) != NULL) return err;
		gpuwrap_addKKT_x_dual(m_pNumCalc, DDf, -1);
		for (IPM_uint i = 0; i < m; i++)
		{
			if ((err = DDinequality(x, DDf, i)) != NULL) return err;
			gpuwrap_addKKT_x_dual(m_pNumCalc, DDf, i);
		}

		if (m > 0)
		{
			gpuwrap_setKKT_Df_i(m_pNumCalc, Df_i);
			gpuwrap_setKKT_f_i(m_pNumCalc);
		}

		if (p > 0)
		{
			gpuwrap_setKKT_A(m_pNumCalc, A);
		}

		/***** calc search direction *****/

		logVector("y", y);
		logVector("r_t", r_t);
		gpuwrap_calcSearchDir(m_pNumCalc, Dy); // kkt and r_t will be corrupted
		logVector("Dy", Dy);

		/***** back tracking line search - from here *****/

		IPM_Scalar s = m_s_coef * gpuwrap_calcMaxScaleBTLS(m_pNumCalc, n, m);

		y_p = y + s * Dy;

		IPM_uint bloop = 0;
		for (; bloop < m_bloop; bloop++)
		{
			// update f_i
			if ((err = inequality(x_p, f_i)) != NULL) return err;

			if ((f_i.maxCoeff() < 0) && (lmd.minCoeff() > 0)) break;
			s = m_beta * s;
			y_p = y + s * Dy;
		}

		IPM_LOG_EN({ if (m_pOuts) *m_pOuts << "s : " << s << endl; });

		if (bloop < m_bloop)
		{
			IPM_LOG_EN({ if (m_pOuts) *m_pOuts << "feasible points found" << endl; });
		}
		else
		{
			IPM_LOG_EN({ if (m_pOuts) *m_pOuts << "infeasible in this direction" << endl; });
		}

		for (; bloop < m_bloop; bloop++)
		{
			// update Df_o, f_i, Df_i
			if ((err = Dobjective(x_p, Df_o)) != NULL) return err;
			if ((err = inequality(x_p, f_i)) != NULL) return err;
			if ((err = Dinequality(x_p, Df_i)) != NULL) return err;

			// update residual
			r_dual = Df_o;
			if (m > 0) r_dual += Df_i.transpose() * lmd_p;
			if (p > 0) r_dual += A.transpose() * nu_p;
			if (m > 0) r_cent = -lmd_p.cwiseProduct(f_i) - inv_t * IPM_Vector::Ones(m);
			if (p > 0) r_pri = A * x_p - b;

			if (r_t.norm() <= (1.0 - m_alpha * s) * org_r_t_norm) break;
			s = m_beta * s;
			y_p = y + s * Dy;
		}

		IPM_LOG_EN({ if (m_pOuts) *m_pOuts << "s : " << s << endl; });

		if ((bloop < m_bloop) && ((y_p - y).norm() >= IPM_SCALAR_EPS))
		{
			IPM_LOG_EN({ if (m_pOuts) *m_pOuts << "update" << endl; });
			// update y
			y = y_p;
		}
		else
		{
			IPM_LOG_EN({ if (m_pOuts) *m_pOuts << "no more improvement" << endl; });
			converged = false;
			break;
		}

		/***** back tracking line search - to here *****/

	} // end of for

	if (!(loop < m_loop))
	{
		IPM_LOG_EN({ if (m_pOuts) *m_pOuts << "iteration limit" << endl; });
		converged = false;
	}


	IPM_LOG_EN({ if (m_pOuts) *m_pOuts << endl << "===== ===== ===== ===== result" << endl; });
	logVector("x", x);
	logVector("lmd", lmd);
	logVector("nu", nu);

	// finalize
	if ((err = finalPoint(x, lmd, nu, converged)) != NULL) return err;

	return NULL;
}
