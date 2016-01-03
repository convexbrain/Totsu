
#ifndef _SOCP_H_
#define _SOCP_H_


#include "PrimalDualIPM.h"


/*!
** @brief A Second-Order Cone Program solver class.
**
** The problem is
** \\[
** \\begin{array}{ll}
** {\\rm minimize} & f^T x \\\\
** {\\rm subject \\ to} & \\| G_i x + h_i \\|_2 \\le c_i^T x + d_i \\quad (i = 0, \\ldots, m - 1) \\\\
** & A x = b,
** \\end{array}
** \\]
** where
** - variables \\( x \\in {\\bf R}^n \\)
** - \\( f \\in {\\bf R}^n \\)
** - \\( G_i \\in {\\bf R}^{n_i \\times n} \\), \\( h_i \\in {\\bf R}^{n_i} \\), \\( c_i \\in {\\bf R}^n \\), \\( d_i \\in {\\bf R} \\)
** - \\( A \\in {\\bf R}^{p \\times n} \\), \\( b \\in {\\bf R}^p \\).
**
** Internally an **approximately equivalent** problem is formed and
** an auxiliary variable \\( s \\in {\\bf R}^m \\) is introduced for the infeasible start method as follows:
** \\[
** \\begin{array}{lll}
** {\\rm minimize}_{x,s} & f^T x \\\\
** {\\rm subject \\ to} & {\\| G_i x + h_i \\|_2^2 \\over s_i} \\le s_i & (i = 0, \\ldots, m - 1) \\\\
** & s_i \\ge \\epsilon_{\\rm bd} & (i = 0, \\ldots, m - 1) \\\\
** & c_i^T x + d_i = s_i & (i = 0, \\ldots, m - 1) \\\\
** & A x = b,
** \\end{array}
** \\]
** where \\( \\epsilon_{\\rm bd} > 0 \\) indicates the extent of approximation that excludes \\( c_i^T x + d_i = 0 \\) boundary.
**/
class SOCP : public PrimalDualIPM
{
public:
	explicit SOCP();
	virtual ~SOCP();

	/*!
	** @brief Runs the solver with given parameters.
	**
	** @param x [IN-OUT] is a column vector containing values of \\(x\\).
	**        It is used as the initial values and overwritten with the final results.
	** @param f [IN] is a column vector containing values of \\(f\\).
	** @param G [IN] is an array of matrices containing values of \\(G_0, \\ldots, G_{m-1}\\).
	** @param h [IN] is an array of vectors containing values of \\(h_0, \\ldots, h_{m-1}\\).
	** @param c [IN] is an array of vectors containing values of \\(c_0, \\ldots, c_{m-1}\\).
	** @param d [IN] is an array of single element matrices containing values of \\(d_0, \\ldots, d_{m-1}\\).
	** @param m is \\(m\\), the number of inequality constraints.
	** @param A [IN] is a matrix containing values of \\(A\\).
	** @param b [IN] is a vector containing values of \\(b\\).
	**/
	IPM_Error solve(IPM_Vector& x,
		const IPM_Vector& f,
		const IPM_Matrix G[], const IPM_Vector h[], const IPM_Vector c[], const IPM_Single d[], const IPM_uint m,
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
	IPM_Scalar m_eps_bd; //!< Value of \\( \\epsilon_{\\rm bd} \\).
	IPM_Scalar m_slack;  //!< Initial margin value for an auxiliary variable.

protected:
	IPM_Vector *m_p_x;

	const IPM_Vector *m_p_f;

	IPM_uint m_m;
	const IPM_Matrix *m_p_G;
	const IPM_Vector *m_p_h;
	const IPM_Vector *m_p_c;
	const IPM_Single *m_p_d;

	const IPM_Matrix *m_p_A;
	const IPM_Vector *m_p_b;

	bool m_converged;
};

#endif // end of ifndef _SOCP_H_
