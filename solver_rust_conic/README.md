# totsu

Totsu ([凸](http://www.decodeunicode.org/en/u+51F8) in Japanese) means convex.

This crate for Rust provides **a first-order conic linear program solver**.

## Target problem

A common target problem is continuous scalar **convex optimization** such as LP, QP, QCQP, SOCP and SDP.
Each of those problems can be represented as a conic linear program.

## Algorithm and design concepts

The author combines the two papers
[\[1\]](https://ieeexplore.ieee.org/abstract/document/6126441)
[\[2\]](https://arxiv.org/abs/1312.3039)
so that the homogeneous self-dual embedding matrix in [\[2\]](https://arxiv.org/abs/1312.3039)
is formed as a linear operator in [\[1\]](https://ieeexplore.ieee.org/abstract/document/6126441).

See [documentation](https://docs.rs/totsu/) for more details.

## Examples
### QP

```rust
use float_eq::assert_float_eq;
use totsu::prelude::*;
use totsu::operator::MatBuild;
use totsu::problem::ProbQP;

type LA = FloatGeneric<f64>;
type AMatBuild = MatBuild<LA, f64>;
type AProbQP = ProbQP<LA, f64>;
type ASolver = Solver<LA, f64>;

let n = 2; // x0, x1
let m = 1;
let p = 0;

// (1/2)(x - a)^2 + const
let mut sym_p = AMatBuild::new(MatType::SymPack(n));
sym_p[(0, 0)] = 1.;
sym_p[(1, 1)] = 1.;

let mut vec_q = AMatBuild::new(MatType::General(n, 1));
vec_q[(0, 0)] = -(-1.); // -a0
vec_q[(1, 0)] = -(-2.); // -a1

// 1 - x0/b0 - x1/b1 <= 0
let mut mat_g = AMatBuild::new(MatType::General(m, n));
mat_g[(0, 0)] = -1. / 2.; // -1/b0
mat_g[(0, 1)] = -1. / 3.; // -1/b1

let mut vec_h = AMatBuild::new(MatType::General(m, 1));
vec_h[(0, 0)] = -1.;

let mat_a = AMatBuild::new(MatType::General(p, n));

let vec_b = AMatBuild::new(MatType::General(p, 1));

let s = ASolver::new().par(|p| {
   p.max_iter = Some(100_000);
});
let mut qp = AProbQP::new(sym_p, vec_q, mat_g, vec_h, mat_a, vec_b, s.par.eps_zero);
let rslt = s.solve(qp.problem(), NullLogger).unwrap();

assert_float_eq!(rslt.0[0..2], [2., 0.].as_ref(), abs_all <= 1e-3);
```

### Other Examples

You can find other [tests](https://github.com/convexbrain/Totsu/tree/master/solver_rust_conic/tests) of pre-defined solvers.
More practical [examples](https://github.com/convexbrain/Totsu/tree/master/examples) are also available.
