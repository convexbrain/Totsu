# totsu_f32cuda

[![Github](https://img.shields.io/github/last-commit/convexbrain/totsu?logo=github)](https://github.com/convexbrain/Totsu)
[![Book](https://img.shields.io/badge/book-日本語-yellow)](https://convexbrain.github.io/Totsu/book/)
[![License](https://img.shields.io/crates/l/totsu.svg)](https://unlicense.org/)

Totsu ([凸](http://www.decodeunicode.org/en/u+51F8) in Japanese) means convex.

This crate for Rust provides **CUDA linear algebra** operations for `totsu`/`totsu_core`.

See [documentation](https://docs.rs/totsu_f32cuda/) for more details.

## Crate family

### totsu_core

[![Crate](https://img.shields.io/crates/v/totsu_core.svg)](https://crates.io/crates/totsu_core)
[![API](https://docs.rs/totsu_core/badge.svg)](https://docs.rs/totsu_core)

A first-order conic linear program solver for convex optimization.

### totsu

[![Crate](https://img.shields.io/crates/v/totsu.svg)](https://crates.io/crates/totsu)
[![API](https://docs.rs/totsu/badge.svg)](https://docs.rs/totsu)

Convex optimization problems LP/QP/QCQP/SOCP/SDP that can be solved by `totsu_core`.

### totsu_f64lapack

[![Crate](https://img.shields.io/crates/v/totsu_f64lapack.svg)](https://crates.io/crates/totsu_f64lapack)
[![API](https://docs.rs/totsu_f64lapack/badge.svg)](https://docs.rs/totsu_f64lapack)

BLAS/LAPACK linear algebra operations for `totsu`/`totsu_core`.

### totsu_f32cuda

[![Crate](https://img.shields.io/crates/v/totsu_f32cuda.svg)](https://crates.io/crates/totsu_f32cuda)
[![API](https://docs.rs/totsu_f32cuda/badge.svg)](https://docs.rs/totsu_f32cuda)

CUDA linear algebra operations for `totsu`/`totsu_core`.
