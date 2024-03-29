use crate::solver::{Cone, SliceMut, SliceLike};
use crate::LinAlgEx;
use num_traits::{Float, cast, ToPrimitive, Zero, One};

//

/// Positive semidefinite cone
/// 
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
/// 
/// \\[
/// {\rm vec}(\mathcal{S}\_+^k) =
/// \left\lbrace x \in \mathbb{R}^n,\ n=\frac12 k(k+1)
/// \ \middle|\ {\rm vec}^{-1}(x) \in \mathcal{S}\_+^k
/// \right\rbrace
/// \\]
/// \\( {\rm vec}(X) = (X_{11}\ \sqrt2 X_{12}\ X_{22}\ \sqrt2 X_{13}\ \sqrt2 X_{23}\ X_{33}\ \cdots)^T \\)
/// which extracts and scales the upper-triangular part of a symmetric matrix X in column-wise.
pub struct ConePSD<'a, L: LinAlgEx>
{
    work: SliceMut<'a, L::Sl>,
    eps_zero: L::F,
}

impl<'a, L: LinAlgEx> ConePSD<'a, L>
{
    /// Query of a length of work slice.
    /// 
    /// Returns the length of work slice that [`ConePSD::new`] requires.
    /// * `nvars` is a number of variables, that is a length of `x` of [`ConePSD::proj`].
    pub fn query_worklen(nvars: usize) -> usize
    {
        let n = (cast::<usize, L::F>(8 * nvars + 1).unwrap().sqrt().to_usize().unwrap() - 1) / 2;
        assert_eq!(n * (n + 1) / 2, nvars);

        L::map_eig_worklen(n)
    }

    /// Creates an instance.
    /// 
    /// Returns the [`ConePSD`] instance.
    /// * `work` slice is used for temporal variables in [`ConePSD::proj`].
    /// * `eps_zero` should be the same value as [`crate::solver::SolverParam::eps_zero`].
    pub fn new(work: &'a mut[L::F], eps_zero: L::F) -> Self
    {
        ConePSD {
            work: L::Sl::new_mut(work),
            eps_zero,
        }
    }
}

impl<'a, L: LinAlgEx> Cone<L> for ConePSD<'a, L>
{
    fn proj(&mut self, _dual_cone: bool, x: &mut L::Sl) -> Result<(), ()>
    {
        if self.work.len() < Self::query_worklen(x.len()) {
            log::error!("work shortage: {} given < {} required", self.work.len(), Self::query_worklen(x.len()));
            return Err(());
        }

        let f0 = L::F::zero();
        let f1 = L::F::one();

        // scale diagonals to match the resulted matrix norm with the vector norm multiplied by 2
        let fsqrt2 = (f1 + f1).sqrt();

        L::map_eig(x, Some(fsqrt2), self.eps_zero, &mut self.work, |e| {
            if e > f0 {
                Some(e)
            }
            else {
                None
            }
        });

        Ok(())
    }

    fn product_group<G: Fn(&mut L::Sl) + Copy>(&self, dp_tau: &mut L::Sl, group: G)
    {
        group(dp_tau);
    }
}

//

#[test]
fn test_cone_psd1()
{
    use float_eq::assert_float_eq;
    use crate::FloatGeneric;

    type L = FloatGeneric<f64>;
    
    let ref_x = &[ // column-major, upper-triangle (seen as if transposed)
        5.,
        0., 0.,
    ];
    let x = &mut[ // column-major, upper-triangle (seen as if transposed)
        5.,
        0., -5.,
    ];
    assert!(ConePSD::<L>::query_worklen(x.len()) <= 10);
    let w = &mut[0.; 10];
    let mut c = ConePSD::<L>::new(w, 1e-12);
    c.proj(false, x).unwrap();
    assert_float_eq!(ref_x, x, abs_all <= 1e-6);
}
