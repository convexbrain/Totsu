# totsu

Totsu ([凸](http://www.decodeunicode.org/en/u+51F8) in Japanese) means convex.

This crate for Rust provides `Solver`, a **first-order conic linear program solver** library.

## Target problem

A common target problem is continuous scalar **convex optimization** such as LP, QP, QCQP, SOCP and SDP.

## Algorithm and design concepts

The author combines the two papers:
1. O’donoghue, Brendan, et al. "Conic optimization via operator splitting and homogeneous self-dual embedding." Journal of Optimization Theory and Applications 169.3 (2016): 1042-1068.
2. Condat, Laurent. "A primal–dual splitting method for convex optimization involving Lipschitzian, proximable and linear composite terms." Journal of Optimization Theory and Applications 158.2 (2013): 460-479.

so that the homogeneous self-dual embedding matrix of the 1st paper is formed as a linear composite term of the 2nd paper's algorithm.

`Solver` has a core method `solve` which takes the following arguments:
* objective and constraint linear operators that implement `Operator` trait and
* a projection onto cones that implements `Cone` trait.

Therefore solving a specific problem requires an implementation of those traits.
You can use a pre-defined implementations (see `predef`), as well as construct a user-defined tailored version for the reason of functionality and efficiency.

*(TODO)*

## Examples
### QP

*(TODO)*

### Other Examples

*(TODO)*
