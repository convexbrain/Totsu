
#ifndef _PRIMAL_DUAL_IPM_H_
#define _PRIMAL_DUAL_IPM_H_

/*! @file
** \brief This header file defines PrimalDualIPM class and other types/macros.
**
** There exist several build configuration macros:
** - \c IPM_DISABLE_LOG disables logging function.
**      This enables to cut dependency on std::ostream.
** - \c IPM_USE_FLOAT uses single floating point precision instead of double.
** - \c IPM_ENABLE_ASSERT enables assertion when generating error string.
**      This is useful when using debugger because it breaks.
**
** These are not defined by default.
*/


#include <Eigen/Dense> // using Eigen 3.2.4 from http://eigen.tuxfamily.org/


#include <cfloat>

#ifdef IPM_USE_FLOAT
#error "!!!!! not implemented yet !!!!!"
// float precision
#define IPM_SCALAR_EPS   FLT_EPSILON
#define IPM_SCALAR_MIN   FLT_MIN
typedef float            IPM_Scalar;
typedef Eigen::VectorXf  IPM_Vector;
typedef Eigen::MatrixXf  IPM_Matrix;
#else
// double precision
#define IPM_SCALAR_EPS   DBL_EPSILON //!< smallest value such that 1 + IPM_SCALAR_EPS != 0
#define IPM_SCALAR_MIN   DBL_MIN     //!< minimum positive value
typedef double           IPM_Scalar; //!< scalar
typedef Eigen::VectorXd  IPM_Vector; //!< vector
typedef Eigen::MatrixXd  IPM_Matrix; //!< matrix
#endif
typedef Eigen::Matrix<IPM_Scalar, 1, 1>     IPM_Single;    //!< single element matrix
typedef const Eigen::Ref<const IPM_Vector>  IPM_Vector_IN; //!< vector input reference/argument
typedef const Eigen::Ref<const IPM_Matrix>  IPM_Matrix_IN; //!< matrix input reference/argument
typedef const Eigen::Ref<const IPM_Single>  IPM_Single_IN; //!< single element matrix input reference/argument
typedef Eigen::Ref<IPM_Vector>              IPM_Vector_IO; //!< vector in-out reference/argument
typedef Eigen::Ref<IPM_Matrix>              IPM_Matrix_IO; //!< matrix in-out reference/argument
typedef Eigen::Ref<IPM_Single>              IPM_Single_IO; //!< single element matrix in-out reference/argument

typedef unsigned int      IPM_uint; //!< unsigned integer

typedef const char*       IPM_Error; //!< error string
#define IPM_NUM_DUMMY(n)  #n
#define IPM_NUM(n)        IPM_NUM_DUMMY(n)

#ifdef IPM_ENABLE_ASSERT
#define IPM_ERR_STR       ( assert(0), __FILE__" @ "IPM_NUM(__LINE__) )
#else
#define IPM_ERR_STR       ( __FILE__ " @ " IPM_NUM(__LINE__) ) //!< generates error string
#endif


class NumCalc_GPU;


/*!
** @brief A basic Primal-Dual Interior-Point Method solver class.
**
** This class abstracts a solver of continuous scalar convex optimization problem:
** \\[
** \\begin{array}{ll}
** {\\rm minimize} & f_{\\rm obj}(x) \\\\
** {\\rm subject \\ to} & f_i(x) \\le 0 \\quad (i = 0, \\ldots, m - 1) \\\\
** & A x = b,
** \\end{array}
** \\]
** where
** - variables \\( x \\in {\\bf R}^n \\)
** - \\( f_{\\rm obj}: {\\bf R}^n \\rightarrow {\\bf R} \\), convex and twice differentiable
** - \\( f_i: {\\bf R}^n \\rightarrow {\\bf R} \\), convex and twice differentiable
** - \\( A \\in {\\bf R}^{p \\times n} \\), \\( {\\bf rank} A = p < n \\), \\( b \\in {\\bf R}^p \\).
**
** The solution gives optimal values of primal variables \\(x\\)
** as well as dual variables \\(\\lambda \\in {\\bf R}^m\\) and \\(\\nu \\in {\\bf R}^p\\).
**/
class PrimalDualIPM
{
public:
	/*!
	** @brief Constructor.
	**/
	explicit PrimalDualIPM();
	/*!
	** @brief Destructor.
	**/
	virtual ~PrimalDualIPM();

	/*!
	** @brief Sets output destination of internal calculation log.
	**
	** @param pOuts is a pointer of output stream. NULL means no output.
	**/
#ifndef IPM_DISABLE_LOG
	void setLog(std::ostream *pOuts) { m_pOuts = pOuts; }
#else
	void setLog(void *pOuts) { }
#endif

protected:
	/*!
	** @brief Returns the larger of a and b.
	**/
	static inline IPM_Scalar max(const IPM_Scalar a, const IPM_Scalar b) { return (a > b) ? a : b; }
	/*!
	** @brief Returns the smaller of a and b.
	**/
	static inline IPM_Scalar min(const IPM_Scalar a, const IPM_Scalar b) { return (a < b) ? a : b; }
	/*!
	** @brief Returns the absolute value of x.
	**/
	static inline IPM_Scalar abs(const IPM_Scalar x) { return (x > 0) ? x : -x; }

	/*!
	** @brief Logs a warning message.
	**
	** @param str is a pointer of the message string.
	** @sa setLog.
	**/
	void logWarn(const char *str);
	/*!
	** @brief Logs values of a vector, transposing if it is a colomn vector.
	**
	** @param str is a pointer of the name string of v.
	** @param v is the logged vector.
	** @sa setLog.
	**/
	void logVector(const char *str, IPM_Vector_IN v);
	/*!
	** @brief Logs values of a matrix.
	**
	** @param str is a pointer of the name string of m.
	** @param m is the logged matrix.
	** @sa setLog.
	**/
	void logMatrix(const char *str, IPM_Matrix_IN m);

	/*!
	** @brief Starts to solve a optimization problem by primal-dual interior-point method.
	**
	** This is the main function of this class.
	** All pure virtual functions are called within this method.
	** @param n is \\(n\\), the dimension of the variable \\(x\\).
	** @param m is \\(m\\), the number of inequality constraints \\(f_i\\).
	** @param p is \\(p\\), the number of rows of equality constraints \\(A\\) and \\(b\\).
	** @return an error string. NULL means no error.
	** @sa initialPoint, finalPoint, objective, Dobjective, DDobjective, inequality, Dinequality, Dinequality, DDinequality and equality.
	**/
	IPM_Error start(const IPM_uint n, const IPM_uint m, const IPM_uint p);

protected:
	/*!
	** @brief Produces the initial values of \\(x\\).
	**
	** **The initial values must satisfy all inequality constraints strictly: \\(f_i(x)<0\\).**
	** This may seem a hard requirement, but introducing **slack variables** helps in many cases.
	** Refer QCQP implementation for example.
	** @param x [OUT] is a column vector containing the initial values of \\(x\\).
	** @return an error string. NULL means no error.
	** @sa inequality.
	**/
	virtual IPM_Error initialPoint(IPM_Vector_IO x) = 0;
	/*!
	** @brief Obtains the final solution values of \\(x\\).
	**
	** Dual variables \\(\\lambda, \\nu\\) can be obtained as well.
	** @param x [IN] is a column vector containing the final values of \\(x\\).
	** @param lmd [IN] is a column vector containing the final values of \\(\\lambda\\).
	** @param nu [IN] is a column vector containing the final values of \\(\\nu\\).
	** @param converged is true when the solution ends with termination criteria satisfied.
	**        false means that x, lmd and nu are not optimal.
	** @return an error string. NULL means no error.
	**/
	virtual IPM_Error finalPoint(IPM_Vector_IN x, IPM_Vector_IN lmd, IPM_Vector_IN nu, bool converged) = 0;
	/*!
	** @brief Calculates the objective function \\(f_{\\rm obj}(x)\\).
	**
	** @param x [IN] is a column vector containing values of \\(x\\).
	** @param f_o [OUT] is a single element matrix containing a value of \\(f_{\\rm obj}(x)\\).
	** @return an error string. NULL means no error.
	** @sa Dobjective, DDobjective.
	**/
	virtual IPM_Error objective(IPM_Vector_IN x, IPM_Single_IO f_o) = 0;
	/*!
	** @brief Calculates first derivatives of the objective function \\(\\nabla f_{\\rm obj}(x)\\).
	**
	** @param x [IN] is a column vector containing values of \\(x\\).
	** @param Df_o [OUT] is a column vector containing values of \\(\\nabla f_{\\rm obj}(x)\\).
	** @return an error string. NULL means no error.
	** @sa objective, DDobjective.
	**/
	virtual IPM_Error Dobjective(IPM_Vector_IN x, IPM_Vector_IO Df_o) = 0;
	/*!
	** @brief Calculates second derivatives of the objective function \\(\\nabla^2 f_{\\rm obj}(x)\\).
	**
	** @param x [IN] is a column vector containing values of \\(x\\).
	** @param DDf_o [OUT] is a matrix containing values of \\(\\nabla^2 f_{\\rm obj}(x)\\).
	** @return an error string. NULL means no error.
	** @sa objective, Dobjective.
	**/
	virtual IPM_Error DDobjective(IPM_Vector_IN x, IPM_Matrix_IO DDf_o) = 0;
	/*!
	** @brief Calculates the inequality constraint functions \\(f_i(x)\\).
	**
	** @param x [IN] is a column vector containing values of \\(x\\).
	** @param f_i [OUT] is a column vector containing values of
	**        \\(\\left[\\matrix{f_0(x) & \\cdots & f_{m-1}(x)}\\right]^T\\).
	** @return an error string. NULL means no error.
	** @sa Dinequality, DDinequality.
	**/
	virtual IPM_Error inequality(IPM_Vector_IN x, IPM_Vector_IO f_i) = 0;
	/*!
	** @brief Calculates first derivatives of the inequality constraint functions \\(Df(x)\\).
	**
	** NOTE: \\(Df(x) = \\left[\\matrix{\\nabla f_0(x) & \\cdots & \\nabla f_{m-1}(x)}\\right]^T\\).
	** @param x [IN] is a column vector containing values of \\(x\\).
	** @param Df_i [OUT] is a matrix containing values of \\(Df(x)\\).
	** @return an error string. NULL means no error.
	** @sa inequality, DDinequality.
	**/
	virtual IPM_Error Dinequality(IPM_Vector_IN x, IPM_Matrix_IO Df_i) = 0;
	/*!
	** @brief Calculates second derivatives of the inequality constraint functions \\(\\nabla^2 f_i(x)\\).
	**
	** @param x [IN] is a column vector containing values of \\(x\\).
	** @param DDf_i [OUT] is a matrix containing values of \\(\\nabla^2 f_i(x)\\).
	** @param of_i indicates an index \\(i\\) of \\(f_i\\).
	** @return an error string. NULL means no error.
	** @sa inequality, Dinequality.
	**/
	virtual IPM_Error DDinequality(IPM_Vector_IN x, IPM_Matrix_IO DDf_i, const IPM_uint of_i) = 0;
	/*!
	** @brief Produces the equality constraints affine parameters \\(A\\) and \\(b\\).
	**
	** @param A [OUT] is a matrix containing values of \\(A\\).
	** @param b [OUT] is a column vector containing values of \\(b\\).
	** @return an error string. NULL means no error.
	**/
	virtual IPM_Error equality(IPM_Matrix_IO A, IPM_Vector_IO b) = 0;

private:
#ifndef IPM_DISABLE_LOG
	std::ostream *m_pOuts;
#endif

	NumCalc_GPU *m_pNumCalc;

protected: // parameters
	IPM_Scalar m_margin; //!< Initial margin value for dual variables of inequalities.

	IPM_Scalar m_loop;  //!< Max iteration number of outer-loop for the Newton step.
	IPM_Scalar m_bloop; //!< Max iteration number of inner-loop for the backtracking line search.

	IPM_Scalar m_eps_feas; //!< Tolerance of the primal and dual residuals.
	IPM_Scalar m_eps;      //!< Tolerance of the surrogate duality gap.

	IPM_Scalar m_mu;    //!< The factor to squeeze complementary slackness.
	IPM_Scalar m_alpha; //!< The factor to decrease residuals in the backtracking line search.
	IPM_Scalar m_beta;  //!< The factor to decrease a step size in the backtracking line search.

	IPM_Scalar m_s_coef; //!< The factor to determine an initial step size in the backtracking line search.
};

#endif // end of ifndef _PRIMAL_DUAL_IPM_H_
