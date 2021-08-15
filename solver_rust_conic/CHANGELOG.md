# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
### Changed
- Bug fix of `ProbSDP`.
### Deprecated
### Removed
### Fixed
### Security

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


[unreleased]: https://github.com/convexbrain/Totsu/compare/rust_conic_v0.8.0...HEAD
[0.8.0]: https://github.com/convexbrain/Totsu/releases/tag/rust_conic_v0.8.0
[0.7.0]: https://github.com/convexbrain/Totsu/releases/tag/rust_conic_v0.7.0
[0.6.0]: https://github.com/convexbrain/Totsu/releases/tag/rust_conic_v0.6.0
[0.5.0]: https://github.com/convexbrain/Totsu/releases/tag/rust_v0.5.0
