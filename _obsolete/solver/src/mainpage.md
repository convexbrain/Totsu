# mainpage

Source code location: https://github.com/convexbrain/Totsu/tree/master/solver/

## Introduction

This C++ package provides a basic **primal-dual interior-point method** solver: PrimalDualIPM.

A common target problem is continuous scalar **convex optimization** such as
LS, LP, QP, GP, QCQP and (approximately equivalent) SOCP.
More specifically,
\\[
\\begin{array}{ll}
{\\rm minimize} & f_{\\rm obj}(x) \\\\
{\\rm subject \\ to} & f_i(x) \\le 0 \\quad (i = 0, \\ldots, m - 1) \\\\
& A x = b,
\\end{array}
\\]
where
* variables \\( x \\in {\\bf R}^n \\)
* \\( f_{\\rm obj}: {\\bf R}^n \\rightarrow {\\bf R} \\), convex and twice differentiable
* \\( f_i: {\\bf R}^n \\rightarrow {\\bf R} \\), convex and twice differentiable
* \\( A \\in {\\bf R}^{p \\times n} \\), \\( b \\in {\\bf R}^p \\).

### Algorithm and design concepts

The overall algorithm is based on the reference:
**S. Boyd and L. Vandenberghe, "Convex Optimization",** http://stanford.edu/~boyd/cvxbook/.

Current version utilizes **Eigen** http://eigen.tuxfamily.org/ for matrix and vector operations.
Other external library is not required.

PrimalDualIPM is written as an abstracted class,
which virtualizes objective and constraint functions (and also those derivatives).
Therefore solving a specific problem requires a derived class that implements those virtual functions.
One can use a pre-defined derived class such as QCQP,
as well as construct a user-defined tailored version for the reason of functionality and efficiency.

## Author

https://twitter.com/convexbrain Any feedback is welcome :-)
