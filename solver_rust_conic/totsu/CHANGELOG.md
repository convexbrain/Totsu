# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Re-export `totsu_core` module.
### Changed
- `Debug` of `MatBuild` prints like `Display`.
- Update version of `totsu_core`.
- Update versions of `float_eq` and `env_logger`.
### Deprecated
### Removed
### Fixed
### Security

## [0.10.1] - 2022-09-24
### Added
- `AsRef` and `AsMut` for `MatBuild`.

## [0.10.0] - 2022-09-23
### Changed
- Totally revised in order to support CUDA linear algebra.
- Divided into 4 crates.

## [0.9.1] - 2022-08-14
### Changed
- Update versions of `log`, `num-traits`, `float_eq` and `intel-mkl-src`.

## [0.9.0] - 2022-01-30
### Added
- Implement `Display` and `Error` trait for `SolverError`.
- `absadd_cols` and `absadd_rows` in `Operator`.
- Trait method `LinAlg::adds`.
### Changed
- Update Rust edition to 2021.
- Update version of `float_eq`.
- Tentative patch for `intel-mkl-src`.
- `Solver` initilization sped-up by using `absadd_cols` and `absadd_rows`.
- `LinAlg::abssum` takes `incx` argument.

## [0.8.1] - 2021-08-17
### Changed
- Update version of `cblas`.
- Update versions of `float_eq` and `env_logger`.
### Fixed
- Bug fix of `ProbSDP`.

## [0.8.0] - 2021-05-30
### Changed
- Using `log` crate.
- Feature `f64lapack` to enable `F64LAPACK`.
- Feature `std` and `libm` for `#![no_std]` support.
### Removed
- `logger` module.
- Feature `nostd`.

## [0.7.0] - 2021-02-13
### Added
- Feature `nostd` for `#![no_std]` support.
### Changed
- `Cone` trait itself doesn't take `eps_zero`.

## [0.6.0] - 2021-02-07
### Changed
- Completely revised from [0.5.0] to conic solver.


[unreleased]: https://github.com/convexbrain/Totsu/compare/totsu_v0.10.1...HEAD
[0.10.1]: https://github.com/convexbrain/Totsu/releases/tag/totsu_v0.10.1
[0.10.0]: https://github.com/convexbrain/Totsu/releases/tag/totsu_v0.10.0
[0.9.1]: https://github.com/convexbrain/Totsu/releases/tag/rust_conic_v0.9.1
[0.9.0]: https://github.com/convexbrain/Totsu/releases/tag/rust_conic_v0.9.0
[0.8.1]: https://github.com/convexbrain/Totsu/releases/tag/rust_conic_v0.8.1
[0.8.0]: https://github.com/convexbrain/Totsu/releases/tag/rust_conic_v0.8.0
[0.7.0]: https://github.com/convexbrain/Totsu/releases/tag/rust_conic_v0.7.0
[0.6.0]: https://github.com/convexbrain/Totsu/releases/tag/rust_conic_v0.6.0
[0.5.0]: https://github.com/convexbrain/Totsu/releases/tag/rust_v0.5.0
