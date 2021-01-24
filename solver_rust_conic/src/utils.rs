use crate::solver::SolverError;

pub(crate) fn split2<T>(
    s: &[T], pos: (usize, usize)
) -> Result<(&[T], &[T]), SolverError>
{
    if s.len() < pos.0 + pos.1 {
        Err(SolverError::WorkShortage)
    }
    else {
        let (s0, spl) = s.split_at(pos.0);
        let (s1, _) = spl.split_at(pos.1);

        Ok((s0, s1))
    }
}

pub(crate) fn split3<T>(
    s: &[T], pos: (usize, usize, usize)
) -> Result<(&[T], &[T], &[T]), SolverError>
{
    if s.len() < pos.0 + pos.1 + pos.2 {
        Err(SolverError::WorkShortage)
    }
    else {
        let (s0, spl) = s.split_at(pos.0);
        let (s1, spl) = spl.split_at(pos.1);
        let (s2, _) = spl.split_at(pos.2);

        Ok((s0, s1, s2))
    }
}

pub(crate) fn split4<T>(
    s: &[T], pos: (usize, usize, usize, usize)
) -> Result<(&[T], &[T], &[T], &[T]), SolverError>
{
    if s.len() < pos.0 + pos.1 + pos.2 + pos.3 {
        Err(SolverError::WorkShortage)
    }
    else {
        let (s0, spl) = s.split_at(pos.0);
        let (s1, spl) = spl.split_at(pos.1);
        let (s2, spl) = spl.split_at(pos.2);
        let (s3, _) = spl.split_at(pos.3);

        Ok((s0, s1, s2, s3))
    }
}

pub(crate) fn split6<T>(
    s: &[T], pos: (usize, usize, usize, usize, usize, usize)
) -> Result<(&[T], &[T], &[T], &[T], &[T], &[T]), SolverError>
{
    if s.len() < pos.0 + pos.1 + pos.2 + pos.3 + pos.4 + pos.5 {
        Err(SolverError::WorkShortage)
    }
    else {
        let (s0, spl) = s.split_at(pos.0);
        let (s1, spl) = spl.split_at(pos.1);
        let (s2, spl) = spl.split_at(pos.2);
        let (s3, spl) = spl.split_at(pos.3);
        let (s4, spl) = spl.split_at(pos.4);
        let (s5, _) = spl.split_at(pos.5);

        Ok((s0, s1, s2, s3, s4, s5))
    }
}

pub(crate) fn split1_mut<T>(
    s: &mut[T], pos: usize
) -> Result<&mut[T], SolverError>
{
    if s.len() < pos {
        Err(SolverError::WorkShortage)
    }
    else {
        let (s0, _) = s.split_at_mut(pos);

        Ok(s0)
    }
}

pub(crate) fn split2_mut<T>(
    s: &mut[T], pos: (usize, usize)
) -> Result<(&mut[T], &mut[T]), SolverError>
{
    if s.len() < pos.0 + pos.1 {
        Err(SolverError::WorkShortage)
    }
    else {
        let (s0, spl) = s.split_at_mut(pos.0);
        let (s1, _) = spl.split_at_mut(pos.1);

        Ok((s0, s1))
    }
}

pub(crate) fn split3_mut<T>(
    s: &mut[T], pos: (usize, usize, usize)
) -> Result<(&mut[T], &mut[T], &mut[T]), SolverError>
{
    if s.len() < pos.0 + pos.1 + pos.2 {
        Err(SolverError::WorkShortage)
    }
    else {
        let (s0, spl) = s.split_at_mut(pos.0);
        let (s1, spl) = spl.split_at_mut(pos.1);
        let (s2, _) = spl.split_at_mut(pos.2);

        Ok((s0, s1, s2))
    }
}

pub(crate) fn split5_mut<T>(
    s: &mut[T], pos: (usize, usize, usize, usize, usize)
) -> Result<(&mut[T], &mut[T], &mut[T], &mut[T], &mut[T]), SolverError>
{
    if s.len() < pos.0 + pos.1 + pos.2 + pos.3 + pos.4 {
        Err(SolverError::WorkShortage)
    }
    else {
        let (s0, spl) = s.split_at_mut(pos.0);
        let (s1, spl) = spl.split_at_mut(pos.1);
        let (s2, spl) = spl.split_at_mut(pos.2);
        let (s3, spl) = spl.split_at_mut(pos.3);
        let (s4, _) = spl.split_at_mut(pos.4);

        Ok((s0, s1, s2, s3, s4))
    }
}

pub(crate) fn split6_mut<T>(
    s: &mut[T], pos: (usize, usize, usize, usize, usize, usize)
) -> Result<(&mut[T], &mut[T], &mut[T], &mut[T], &mut[T], &mut[T]), SolverError>
{
    if s.len() < pos.0 + pos.1 + pos.2 + pos.3 + pos.4 + pos.5 {
        Err(SolverError::WorkShortage)
    }
    else {
        let (s0, spl) = s.split_at_mut(pos.0);
        let (s1, spl) = spl.split_at_mut(pos.1);
        let (s2, spl) = spl.split_at_mut(pos.2);
        let (s3, spl) = spl.split_at_mut(pos.3);
        let (s4, spl) = spl.split_at_mut(pos.4);
        let (s5, _) = spl.split_at_mut(pos.5);

        Ok((s0, s1, s2, s3, s4, s5))
    }
}

pub(crate) fn split7_mut<T>(
    s: &mut[T], pos: (usize, usize, usize, usize, usize, usize, usize)
) -> Result<(&mut[T], &mut[T], &mut[T], &mut[T], &mut[T], &mut[T], &mut[T]), SolverError>
{
    if s.len() < pos.0 + pos.1 + pos.2 + pos.3 + pos.4 + pos.5 + pos.6 {
        Err(SolverError::WorkShortage)
    }
    else {
        let (s0, spl) = s.split_at_mut(pos.0);
        let (s1, spl) = spl.split_at_mut(pos.1);
        let (s2, spl) = spl.split_at_mut(pos.2);
        let (s3, spl) = spl.split_at_mut(pos.3);
        let (s4, spl) = spl.split_at_mut(pos.4);
        let (s5, spl) = spl.split_at_mut(pos.5);
        let (s6, _) = spl.split_at_mut(pos.6);

        Ok((s0, s1, s2, s3, s4, s5, s6))
    }
}
