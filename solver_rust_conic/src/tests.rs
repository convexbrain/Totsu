extern crate intel_mkl_src;

use crate::solver::Solver;

#[test]
fn test_smoke() {
    let s = Solver::new();
    s.solve();
}
