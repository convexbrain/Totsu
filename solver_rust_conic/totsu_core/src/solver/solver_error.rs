/// Solver errors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SolverError
{
    /// Found an unbounded certificate.
    Unbounded,
    /// Found an infeasibile certificate.
    Infeasible,
    /// Exceed max iterations.
    ExcessIter,

    /// Invalid [`crate::solver::Operator`].
    InvalidOp,
    /// Shortage of work slice length.
    WorkShortage,
    /// Failure caused by [`crate::solver::Cone`].
    ConeFailure,
}

impl core::fmt::Display for SolverError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", match &self {
            SolverError::Unbounded    => "Unbounded: found an unbounded certificate",
            SolverError::Infeasible   => "Infeasible: found an infeasibile certificate",
            SolverError::ExcessIter   => "ExcessIter: exceed max iterations",
            SolverError::InvalidOp    => "InvalidOp: invalid Operator",
            SolverError::WorkShortage => "WorkShortage: shortage of work slice length",
            SolverError::ConeFailure  => "ConeFailure: failure caused by Cone",
        })
    }
}

//

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "std")]
impl std::error::Error for SolverError {}
