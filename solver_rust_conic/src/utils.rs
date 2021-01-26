pub(crate) fn split2<T>(
    s: &[T], pos: (usize, usize)
) -> Option<(&[T], &[T])>
{
    if s.len() < pos.0 + pos.1 {
        None
    }
    else {
        let (s0, spl) = s.split_at(pos.0);
        let (s1, _) = spl.split_at(pos.1);

        Some((s0, s1))
    }
}

pub(crate) fn split3<T>(
    s: &[T], pos: (usize, usize, usize)
) -> Option<(&[T], &[T], &[T])>
{
    if s.len() < pos.0 + pos.1 + pos.2 {
        None
    }
    else {
        let (s0, spl) = s.split_at(pos.0);
        let (s1, spl) = spl.split_at(pos.1);
        let (s2, _) = spl.split_at(pos.2);

        Some((s0, s1, s2))
    }
}

pub(crate) fn split4<T>(
    s: &[T], pos: (usize, usize, usize, usize)
) -> Option<(&[T], &[T], &[T], &[T])>
{
    if s.len() < pos.0 + pos.1 + pos.2 + pos.3 {
        None
    }
    else {
        let (s0, spl) = s.split_at(pos.0);
        let (s1, spl) = spl.split_at(pos.1);
        let (s2, spl) = spl.split_at(pos.2);
        let (s3, _) = spl.split_at(pos.3);

        Some((s0, s1, s2, s3))
    }
}

pub(crate) fn split6<T>(
    s: &[T], pos: (usize, usize, usize, usize, usize, usize)
) -> Option<(&[T], &[T], &[T], &[T], &[T], &[T])>
{
    if s.len() < pos.0 + pos.1 + pos.2 + pos.3 + pos.4 + pos.5 {
        None
    }
    else {
        let (s0, spl) = s.split_at(pos.0);
        let (s1, spl) = spl.split_at(pos.1);
        let (s2, spl) = spl.split_at(pos.2);
        let (s3, spl) = spl.split_at(pos.3);
        let (s4, spl) = spl.split_at(pos.4);
        let (s5, _) = spl.split_at(pos.5);

        Some((s0, s1, s2, s3, s4, s5))
    }
}

pub(crate) fn split1_mut<T>(
    s: &mut[T], pos: usize
) -> Option<&mut[T]>
{
    if s.len() < pos {
        None
    }
    else {
        let (s0, _) = s.split_at_mut(pos);

        Some(s0)
    }
}

pub(crate) fn split2_mut<T>(
    s: &mut[T], pos: (usize, usize)
) -> Option<(&mut[T], &mut[T])>
{
    if s.len() < pos.0 + pos.1 {
        None
    }
    else {
        let (s0, spl) = s.split_at_mut(pos.0);
        let (s1, _) = spl.split_at_mut(pos.1);

        Some((s0, s1))
    }
}

pub(crate) fn split3_mut<T>(
    s: &mut[T], pos: (usize, usize, usize)
) -> Option<(&mut[T], &mut[T], &mut[T])>
{
    if s.len() < pos.0 + pos.1 + pos.2 {
        None
    }
    else {
        let (s0, spl) = s.split_at_mut(pos.0);
        let (s1, spl) = spl.split_at_mut(pos.1);
        let (s2, _) = spl.split_at_mut(pos.2);

        Some((s0, s1, s2))
    }
}

pub(crate) fn split5_mut<T>(
    s: &mut[T], pos: (usize, usize, usize, usize, usize)
) -> Option<(&mut[T], &mut[T], &mut[T], &mut[T], &mut[T])>
{
    if s.len() < pos.0 + pos.1 + pos.2 + pos.3 + pos.4 {
        None
    }
    else {
        let (s0, spl) = s.split_at_mut(pos.0);
        let (s1, spl) = spl.split_at_mut(pos.1);
        let (s2, spl) = spl.split_at_mut(pos.2);
        let (s3, spl) = spl.split_at_mut(pos.3);
        let (s4, _) = spl.split_at_mut(pos.4);

        Some((s0, s1, s2, s3, s4))
    }
}

pub(crate) fn split6_mut<T>(
    s: &mut[T], pos: (usize, usize, usize, usize, usize, usize)
) -> Option<(&mut[T], &mut[T], &mut[T], &mut[T], &mut[T], &mut[T])>
{
    if s.len() < pos.0 + pos.1 + pos.2 + pos.3 + pos.4 + pos.5 {
        None
    }
    else {
        let (s0, spl) = s.split_at_mut(pos.0);
        let (s1, spl) = spl.split_at_mut(pos.1);
        let (s2, spl) = spl.split_at_mut(pos.2);
        let (s3, spl) = spl.split_at_mut(pos.3);
        let (s4, spl) = spl.split_at_mut(pos.4);
        let (s5, _) = spl.split_at_mut(pos.5);

        Some((s0, s1, s2, s3, s4, s5))
    }
}
