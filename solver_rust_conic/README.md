# totsu

Totsu ([凸](http://www.decodeunicode.org/en/u+51F8) in Japanese) means convex.

This crate for Rust provides `Solver`, a **first-order conic linear program solver** library.

## Target problem

A common target problem is continuous scalar **convex optimization** such as LP, QP, QCQP, SOCP and SDP.

## Algorithm and design concepts

The author combines the two papers [1][2] so that the homogeneous self-dual embedding matrix in [1] is formed as a linear operator in [2].

`Solver` has a core method `solve` which takes the following arguments:
* objective and constraint linear operators that implement `Operator` trait and
* a projection onto cones that implements `Cone` trait.

Therefore solving a specific problem requires an implementation of those traits.
You can use a pre-defined implementations (see `problem`), as well as construct a user-defined tailored version for the reason of functionality and efficiency.

*(TODO)*

## Examples
### QP

*(TODO)*

### Other Examples

*(TODO)*

## References

1. O’donoghue, Brendan, et al. "Conic optimization via operator splitting and homogeneous self-dual embedding." Journal of Optimization Theory and Applications 169.3 (2016): 1042-1068.
1. Pock, Thomas, and Antonin Chambolle. "Diagonal preconditioning for first order primal-dual algorithms in convex optimization." 2011 International Conference on Computer Vision. IEEE, 2011.
1. Parikh, Neal, and Stephen Boyd. "Proximal algorithms." Foundations and Trends in optimization 1.3 (2014): 127-239.
1. ApS, Mosek. "MOSEK modeling cookbook." (2020).
1. Andersen, Martin, et al. "Interior-point methods for large-scale cone programming." Optimization for machine learning 5583 (2011).
1. Boyd, Stephen, and Lieven Vandenberghe. "Convex Optimization." (2004).
