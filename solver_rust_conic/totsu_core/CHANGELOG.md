# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `LinAlgEx::add_sub`, `LinAlg::max`, `LinAlg::min` for performance improvement.
### Changed
- Performance improvement of `ConeRotSOC::proj`, `ConeRPos::proj`.
- Performance improvement of `SliceLike::get` and `SliceLike::set`.
- Performance improvement of `Solver::solve`.
### Deprecated
### Removed
### Fixed
### Security

## [0.1.0] - 2022-09-23
### Changed
- Totally revised in order to support CUDA linear algebra.
- Divided into 4 crates.


[unreleased]: https://github.com/convexbrain/Totsu/compare/totsu_core_v0.1.0...HEAD
[0.1.0]: https://github.com/convexbrain/Totsu/releases/tag/totsu_core_v0.1.0
