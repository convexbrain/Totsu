use totsu::totsu_core::{ConeRPos, ConeSOC};
use totsu::totsu_core::solver::Cone;

use super::La;

//

pub struct ProbCone
{
    x_sz: usize,
    t_sz: usize,
    rpos: ConeRPos<La>,
    soc: ConeSOC<La>,
}

impl ProbCone
{
    pub fn new(width: usize, height: usize) -> Self
    {
        ProbCone {
            x_sz: width * height,
            t_sz: (width - 2) * (height - 2),
            rpos: ConeRPos::new(),
            soc: ConeSOC::new(),
        }
    }
}

impl Cone<La> for ProbCone
{
    fn proj(&mut self, dual_cone: bool, x: &mut[f64]) -> Result<(), ()>
    {
        let (x_rpos, x_soc) = x.split_at_mut(self.t_sz * 2 + 1 + self.x_sz * 2);

        self.rpos.proj(dual_cone, x_rpos)?;
        self.soc.proj(dual_cone, x_soc)?;

        Ok(())
    }

    fn product_group<G: Fn(&mut[f64]) + Copy>(&self, dp_tau: &mut[f64], group: G)
    {
        let (dp_tau_rpos, dp_tau_soc) = dp_tau.split_at_mut(self.t_sz * 2 + 1 + self.x_sz * 2);

        self.rpos.product_group(dp_tau_rpos, group);
        self.soc.product_group(dp_tau_soc, group);
    }
}
