# totsu

Totsu ([凸](http://www.decodeunicode.org/en/u+51F8) in Japanese) means convex.

This crate for Rust provides a basic **primal-dual interior-point method** solver: `PDIPM`.

## Target problem

A common target problem is continuous scalar **convex optimization** such as
LP, QP and QCQP. SOCP and SDP can also be handled with a certain effort.

## Algorithm and design concepts

The overall algorithm is based on the reference:
*S. Boyd and L. Vandenberghe, "Convex Optimization",*
[http://stanford.edu/~boyd/cvxbook/](http://stanford.edu/~boyd/cvxbook/).

`PDIPM` has a core method `solve`
which takes objective and constraint (derivative) functions as closures.
Therefore solving a specific problem requires an implementation of those closures.
You can use a pre-defined implementations (see `predef`),
as well as construct a user-defined tailored version for the reason of functionality and efficiency.

This crate has no dependencies on other crates at all.
Necessary matrix operations are implemented in `mat` and `matsvd`.

## Examples
### QP

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

let param = PDIPMParam::default();
let rslt = PDIPM::new().solve_qp(&param, &mut std::io::sink(),
                                 &mat_p, &vec_q,
                                 &mat_g, &vec_h,
                                 &mat_a, &vec_b).unwrap();

let exp = Mat::new_vec(n).set_iter(&[
    2., 0.
]);
println!("rslt = {}", rslt);
assert!((&rslt - exp).norm_p2() < param.eps);
```

### Other Examples

You can find other test examples of pre-defined solvers in `lib.rs`.
More practical examples are available [here](https://github.com/convexbrain/Totsu/tree/master/examples).
