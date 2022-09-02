// TODO
#[macro_export]
macro_rules! splitm {
    ($slice:expr, $( ($var:ident; $len:expr) ),+ ) => {
        let (_, _splitm_rest) = $slice.split_at(0);
        $(
            let ($var, _splitm_rest) = _splitm_rest.split_at($len);
        )*
        drop(_splitm_rest);
    };
}

// TODO
#[macro_export]
macro_rules! splitm_mut {
    ($slice:expr, $( ($var:ident; $len:expr) ),+ ) => {
        let (_, mut _splitm_rest) = $slice.split_at_mut(0);
        $(
            let (mut $var, mut _splitm_rest) = _splitm_rest.split_at_mut($len);
        )*
        drop(_splitm_rest);
    };
}

pub trait SplitN
where Self: core::marker::Sized
{
    fn split1(self, p0: usize)
    -> Option<Self>;

    fn split2(self, p0: usize, p1: usize)
    -> Option<(Self, Self)>;

    fn split3(self, p0: usize, p1: usize, p2: usize)
    -> Option<(Self, Self, Self)>;

    fn split4(self, p0: usize, p1: usize, p2: usize, p3: usize)
    -> Option<(Self, Self, Self, Self)>;

    fn split5(self, p0: usize, p1: usize, p2: usize, p3: usize, p4: usize)
    -> Option<(Self, Self, Self, Self, Self)>;

    fn split6(self, p0: usize, p1: usize, p2: usize, p3: usize, p4: usize, p5: usize)
    -> Option<(Self, Self, Self, Self, Self, Self)>;
}

impl<T> SplitN for &[T]
{
    fn split1(self, p0: usize)
    -> Option<Self>
    {
        if self.len() < p0 {
            None
        }
        else {
            let (__, spl) = self.split_at(0);
            let (s0, ___) = spl.split_at(p0);
    
            Some(s0)
        }
    }

    fn split2(self, p0: usize, p1: usize)
    -> Option<(Self, Self)>
    {
        if self.len() < p0 + p1 {
            None
        }
        else {
            let (__, spl) = self.split_at(0);
            let (s0, spl) = spl.split_at(p0);
            let (s1, ___) = spl.split_at(p1);
    
            Some((s0, s1))
        }
    }

    fn split3(self, p0: usize, p1: usize, p2: usize)
    -> Option<(Self, Self, Self)>
    {
        if self.len() < p0 + p1 + p2 {
            None
        }
        else {
            let (__, spl) = self.split_at(0);
            let (s0, spl) = spl.split_at(p0);
            let (s1, spl) = spl.split_at(p1);
            let (s2, ___) = spl.split_at(p2);
    
            Some((s0, s1, s2))
        }
    }

    fn split4(self, p0: usize, p1: usize, p2: usize, p3: usize)
    -> Option<(Self, Self, Self, Self)>
    {
        if self.len() < p0 + p1 + p2 + p3 {
            None
        }
        else {
            let (__, spl) = self.split_at(0);
            let (s0, spl) = spl.split_at(p0);
            let (s1, spl) = spl.split_at(p1);
            let (s2, spl) = spl.split_at(p2);
            let (s3, ___) = spl.split_at(p3);
    
            Some((s0, s1, s2, s3))
        }
    }

    fn split5(self, p0: usize, p1: usize, p2: usize, p3: usize, p4: usize)
    -> Option<(Self, Self, Self, Self, Self)>
    {
        if self.len() < p0 + p1 + p2 + p3 + p4 {
            None
        }
        else {
            let (__, spl) = self.split_at(0);
            let (s0, spl) = spl.split_at(p0);
            let (s1, spl) = spl.split_at(p1);
            let (s2, spl) = spl.split_at(p2);
            let (s3, spl) = spl.split_at(p3);
            let (s4, ___) = spl.split_at(p4);
    
            Some((s0, s1, s2, s3, s4))
        }
    }

    fn split6(self, p0: usize, p1: usize, p2: usize, p3: usize, p4: usize, p5: usize)
    -> Option<(Self, Self, Self, Self, Self, Self)>
    {
        if self.len() < p0 + p1 + p2 + p3 + p4 + p5 {
            None
        }
        else {
            let (__, spl) = self.split_at(0);
            let (s0, spl) = spl.split_at(p0);
            let (s1, spl) = spl.split_at(p1);
            let (s2, spl) = spl.split_at(p2);
            let (s3, spl) = spl.split_at(p3);
            let (s4, spl) = spl.split_at(p4);
            let (s5, ___) = spl.split_at(p5);
    
            Some((s0, s1, s2, s3, s4, s5))
        }
    }
}

impl<T> SplitN for &mut[T]
{
    fn split1(self, p0: usize)
    -> Option<Self>
    {
        if self.len() < p0 {
            None
        }
        else {
            let (__, spl) = self.split_at_mut(0);
            let (s0, ___) = spl.split_at_mut(p0);
    
            Some(s0)
        }
    }

    fn split2(self, p0: usize, p1: usize)
    -> Option<(Self, Self)>
    {
        if self.len() < p0 + p1 {
            None
        }
        else {
            let (__, spl) = self.split_at_mut(0);
            let (s0, spl) = spl.split_at_mut(p0);
            let (s1, ___) = spl.split_at_mut(p1);
    
            Some((s0, s1))
        }
    }

    fn split3(self, p0: usize, p1: usize, p2: usize)
    -> Option<(Self, Self, Self)>
    {
        if self.len() < p0 + p1 + p2 {
            None
        }
        else {
            let (__, spl) = self.split_at_mut(0);
            let (s0, spl) = spl.split_at_mut(p0);
            let (s1, spl) = spl.split_at_mut(p1);
            let (s2, ___) = spl.split_at_mut(p2);
    
            Some((s0, s1, s2))
        }
    }

    fn split4(self, p0: usize, p1: usize, p2: usize, p3: usize)
    -> Option<(Self, Self, Self, Self)>
    {
        if self.len() < p0 + p1 + p2 + p3 {
            None
        }
        else {
            let (__, spl) = self.split_at_mut(0);
            let (s0, spl) = spl.split_at_mut(p0);
            let (s1, spl) = spl.split_at_mut(p1);
            let (s2, spl) = spl.split_at_mut(p2);
            let (s3, ___) = spl.split_at_mut(p3);
    
            Some((s0, s1, s2, s3))
        }
    }

    fn split5(self, p0: usize, p1: usize, p2: usize, p3: usize, p4: usize)
    -> Option<(Self, Self, Self, Self, Self)>
    {
        if self.len() < p0 + p1 + p2 + p3 + p4 {
            None
        }
        else {
            let (__, spl) = self.split_at_mut(0);
            let (s0, spl) = spl.split_at_mut(p0);
            let (s1, spl) = spl.split_at_mut(p1);
            let (s2, spl) = spl.split_at_mut(p2);
            let (s3, spl) = spl.split_at_mut(p3);
            let (s4, ___) = spl.split_at_mut(p4);
    
            Some((s0, s1, s2, s3, s4))
        }
    }

    fn split6(self, p0: usize, p1: usize, p2: usize, p3: usize, p4: usize, p5: usize)
    -> Option<(Self, Self, Self, Self, Self, Self)>
    {
        if self.len() < p0 + p1 + p2 + p3 + p4 + p5 {
            None
        }
        else {
            let (__, spl) = self.split_at_mut(0);
            let (s0, spl) = spl.split_at_mut(p0);
            let (s1, spl) = spl.split_at_mut(p1);
            let (s2, spl) = spl.split_at_mut(p2);
            let (s3, spl) = spl.split_at_mut(p3);
            let (s4, spl) = spl.split_at_mut(p4);
            let (s5, ___) = spl.split_at_mut(p5);
    
            Some((s0, s1, s2, s3, s4, s5))
        }
    }
}
