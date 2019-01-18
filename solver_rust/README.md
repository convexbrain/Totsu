# totsu

Totsu ([凸](http://www.decodeunicode.org/en/u+51F8) in Japanese) means convex.

<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML' async></script>

This crate provides a basic **primal-dual interior-point method** solver: [`PDIPM`](pdipm/struct.PDIPM.html).

## Target problem

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

## Algorithm and design concepts

The overall algorithm is based on the reference:
*S. Boyd and L. Vandenberghe, "Convex Optimization",*
[http://stanford.edu/~boyd/cvxbook/](http://stanford.edu/~boyd/cvxbook/).

[`PDIPM`](pdipm/struct.PDIPM.html) has a core method [`solve`](pdipm/struct.PDIPM.html#method.solve)
which takes objective and constraint (derivative) functions as closures.
Therefore solving a specific problem requires a implementation of those closures.
You can use a pre-defined implementations (see [`predef`](predef/index.html)),
as well as construct a user-defined tailored version for the reason of functionality and efficiency.

This crate has no dependencies on other crates at all.
Necessary matrix operations are implemented in [`mat`](mat/index.html) and [`matsvd`](matsvd/index.html).

## Example: QP

```rust
use totsu::prelude::*;
use totsu::predef::*;

let n: usize = 2; // x0, x1
let m: usize = 1;
let p: usize = 0;

// (1/2)(x - a)^2 + const
let mat_p = Mat::new(n, n).set_iter(&[
    1., 0.,
    0., 1.
]);
let vec_q = Mat::new_vec(n).set_iter(&[
    -(-1.), // -a0
    -(-2.)  // -a1
]);

// 1 - x0/b0 - x1/b1 <= 0
let mat_g = Mat::new(m, n).set_iter(&[
    -1. / 2., // -1/b0
    -1. / 3.  // -1/b1
]);
let vec_h = Mat::new_vec(m).set_iter(&[
    -1.
]);

let mat_a = Mat::new(p, n);
let vec_b = Mat::new_vec(p);

let pdipm = PDIPM::new();
let rslt = pdipm.solve_qp(std::io::sink(),
                          &mat_p, &vec_q,
                          &mat_g, &vec_h,
                          &mat_a, &vec_b).unwrap();

let exp = Mat::new_vec(n).set_iter(&[
    2., 0.
]);
assert!((&rslt - exp).norm_p2() < pdipm.eps, "rslt = {}", rslt);
```

License: MIT
