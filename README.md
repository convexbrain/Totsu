# Totsu

Totsu (凸 in Japanese) means convex.

## solver_rust_conic/

The crates for Rust of

* `totsu_core`: A first-order conic linear program solver for convex optimization.
* `totsu`: Convex optimization problems LP/QP/QCQP/SOCP/SDP that can be solved by `totsu_core`.
* `totsu_f64lapack`: BLAS/LAPACK linear algebra operations for `totsu`/`totsu_core`.
* `totsu_f32cuda`: CUDA linear algebra operations for `totsu`/`totsu_core`.

Book(ja): https://convexbrain.github.io/Totsu/book/

## examples/

How to use Totsu to solve various optimization problems.

---

## [obsolete] solver/

This C++ package provides a basic **primal-dual interior-point method** solver: `PrimalDualIPM`.

solver/doxygen/ documentation: http://convexbrain.github.io/Totsu/PrimalDualIPM/html/

## [obsolete] solver_rust/

This crate for Rust provides a basic **primal-dual interior-point method** solver: `PDIPM`.

Crate: https://crates.io/crates/totsu/0.5.0/

Documentation: https://docs.rs/totsu/0.5.0/

