use super::*;

// used by examples

impl<L: LinAlgEx> core::fmt::Debug for MatBuild<L>
where L::F: Float + core::fmt::LowerExp
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> Result<(), core::fmt::Error>
    {
        let (nr, nc) = self.size();

        if nr == 0 || nc == 0 {
            write!(f, "[ ]")?;
        }
        else {
            for r in 0..nr {
                if r == 0 {
                    write!(f, "[")?;
                }
                else {
                    write!(f, " ")?;
                }

                for c in 0..nc {
                    write!(f, " {:.3e}", self[(r, c)])?;
                }

                if r < nr - 1 {
                    writeln!(f)?;
                }
                else {
                    write!(f, " ] ({} x {}) ", nr, nc)?;
                    match self.typ {
                        MatType::General(_, _) => write!(f, "General")?,
                        MatType::SymPack(_) => write!(f, "Symmetric Packed")?,
                    }
                }
            }
        }

        Ok(())
    }
}

impl<L: LinAlgEx> core::fmt::Display for MatBuild<L>
where L::F: Float + core::fmt::LowerExp
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> Result<(), core::fmt::Error>
    {
        let (nr, nc) = self.size();
        if nr == 0 || nc == 0 {
            write!(f, "[ ]")?;
        }
        else {
            write!(f, "[ {:.3e}", self[(0, 0)])?;
            if nc > 2 {
                write!(f, " ...")?;
            }
            if nc > 1 {
                write!(f, " {:.3e}", self[(0, nc - 1)])?;
            }

            if nr > 2 {
                writeln!(f)?;
                write!(f, "  ...")?;
            }

            if nr > 1 {
                writeln!(f)?;
                write!(f, "  {:.3e}", self[(nr - 1, 0)])?;
                if nc > 2 {
                    write!(f, " ...")?;
                }
                if nc > 1 {
                    write!(f, " {:.3e}", self[(nr - 1, nc - 1)])?;
                }
            }
            write!(f, " ]")?;
        }

        write!(f, " ({} x {}) ", nr, nc)?;
        match self.typ {
            MatType::General(_, _) => write!(f, "General")?,
            MatType::SymPack(_) => write!(f, "Symmetric Packed")?,
        }

        Ok(())
    }
}

impl<L: LinAlgEx> AsRef<[L::F]> for MatBuild<L>
{
    fn as_ref(&self) -> &[L::F]
    {
        &self.array
    }
}

impl<L: LinAlgEx> AsMut<[L::F]> for MatBuild<L>
{
    fn as_mut(&mut self) -> &mut[L::F]
    {
        &mut self.array
    }
}
