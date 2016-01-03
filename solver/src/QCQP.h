
#ifndef _QCQP_H_
#define _QCQP_H_


#include "PrimalDualIPM.h"


/*!
** @brief A Quadratically Constrained Quadratic Program solver class.
**
** The problem is
** \\[
** \\begin{array}{ll}
** {\\rm minimize} & {1 \\over 2} x^T P_0 x + q_0^T x + r_0 \\\\
** {\\rm subject \\ to} & {1 \\over 2} x^T P_i x + q_i^T x + r_i \\le 0 \\quad (i = 1, \\ldots, m) \\\\
** & A x = b,
** \\end{array}
** \\]
** where
** - variables \\( x \\in {\\bf R}^n \\)
** - \\( P_j \\in {\\bf S}_{+}^n \\), \\( q_j \\in {\\bf R}^n \\), \\( r_j \\in {\\bf R} \\) for \\( j = 0, \\ldots, m \\)
** - \\( A \\in {\\bf R}^{p \\times n} \\), \\( b \\in {\\bf R}^p \\).
**
** Internally a slack variable \\( s \\in {\\bf R} \\) is introduced for the infeasible start method as follows:
** \\[
** \\begin{array}{ll}
** {\\rm minimize}_{x,s} & {1 \\over 2} x^T P_0 x + q_0^T x + r_0 \\\\
** {\\rm subject \\ to} & {1 \\over 2} x^T P_i x + q_i^T x + r_i \\le s \\quad (i = 1, \\ldots, m) \\\\
** & A x = b \\\\
** & s = 0.
** \\end{array}
** \\]
**/
class QCQP : public PrimalDualIPM
{
public:
	explicit QCQP();
	virtual ~QCQP();

	/*!
	** @brief Runs the solver with given parameters.
	**
	** @param x [IN-OUT] is a column vector containing values of \\(x\\).
	**        It is used as the initial values and overwritten with the final results.
	** @param P [IN] is an array of matrices containing values of \\(P_0, \\ldots, P_m\\).
	** @param q [IN] is an array of vectors containing values of \\(q_0, \\ldots, q_m\\).
	** @param r [IN] is an array of single element matrices containing values of \\(r_0, \\ldots, r_m\\).
	** @param m is \\(m\\), the number of inequality constraints.
	** @param A [IN] is a matrix containing values of \\(A\\).
	** @param b [IN] is a vector containing values of \\(b\\).
	**/
	IPM_Error solve(IPM_Vector& x,
					const IPM_Matrix P[], const IPM_Vector q[], const IPM_Single r[], const IPM_uint m,
					const IPM_Matrix& A, const IPM_Vector& b);

	/*!
	** @brief Indicates if the previous solve() has converged or not.
	**/
	bool isConverged(void) { return m_converged; }

protected:
	virtual IPM_Error initialPoint(IPM_Vector_IO x);
	virtual IPM_Error finalPoint(IPM_Vector_IN x, IPM_Vector_IN lmd, IPM_Vector_IN nu, bool converged);
	virtual IPM_Error objective(IPM_Vector_IN x, IPM_Single_IO f_o);
	virtual IPM_Error Dobjective(IPM_Vector_IN x, IPM_Vector_IO Df_o);
	virtual IPM_Error DDobjective(IPM_Vector_IN x, IPM_Matrix_IO DDf_o);
	virtual IPM_Error inequality(IPM_Vector_IN x, IPM_Vector_IO f_i);
	virtual IPM_Error Dinequality(IPM_Vector_IN x, IPM_Matrix_IO Df_i);
	virtual IPM_Error DDinequality(IPM_Vector_IN x, IPM_Matrix_IO DDf_i, const IPM_uint of_i);
	virtual IPM_Error equality(IPM_Matrix_IO A, IPM_Vector_IO b);

protected:
	IPM_Scalar m_slack; //!< Initial margin value for a slack variable.

protected:
	IPM_Vector *m_p_x;

	IPM_uint m_m;
	const IPM_Matrix *m_p_P;
	const IPM_Vector *m_p_q;
	const IPM_Single *m_p_r;

	const IPM_Matrix *m_p_A;
	const IPM_Vector *m_p_b;

	bool m_converged;
};

#endif // end of ifndef _QCQP_H_
