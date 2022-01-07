use plotters::prelude::*;
use std::marker::PhantomData;
use num_traits::Float;

pub struct ContourSeries<DB: DrawingBackend, V, F>
where F: Fn(V, V) -> V
{
    style: ShapeStyle,
    data_x: Vec<V>,
    data_y: Vec<V>,
    thr: V,
    func_f: F,
    phantom: PhantomData<DB>,
    idx_x: usize,
    idx_y: usize,
}

fn bilinear<V: Float>(x0: V, x1: V, z0: V, z1: V) -> V
{
    let v0 = V::zero();
    let v1 = V::one();
    
    assert_ne!(z0 < v0, z1 < v0);

    let z = z0.abs() + z1.abs();
    let alpha = if z > v0 {z1.abs() / z} else {v1 / (v1 + v1)};
    alpha * x0 + (v1 - alpha) * x1
}

impl<DB: DrawingBackend, V, F> ContourSeries<DB, V, F>
where V: Float, F: Fn(V, V) -> V
{
    pub fn new<IX: IntoIterator<Item = V>, IY: IntoIterator<Item = V>, S: Into<ShapeStyle>>(
        x_iter: IX, y_iter: IY, func_f: F, style: S
    ) -> Self
    {
        Self {
            style: style.into(),
            data_x: x_iter.into_iter().collect(),
            data_y: y_iter.into_iter().collect(),
            thr: V::zero(),
            func_f,
            phantom: PhantomData,
            idx_x: 0,
            idx_y: 0,
        }
    }

    fn next_step(&mut self) -> Option<(usize, usize)>
    {
        loop {
            if self.idx_y + 1 >= self.data_y.len() {
                return None;
            }
            else {
                if self.idx_x + 1 >= self.data_x.len() {
                    self.idx_x = 0;
                    self.idx_y += 1;
                    continue;
                }
                else {
                    let ix = self.idx_x;
                    self.idx_x += 1;
                    return Some((ix, self.idx_y));
                }
            }
        }
    }

    fn next_line(&mut self) -> Result<((V, V), (V, V)), bool>
    {
        let v0 = V::zero();
        
        if let Some((ix, iy)) = self.next_step() {
            let x0 = self.data_x[ix];
            let x1 = self.data_x[ix + 1];
            let y0 = self.data_y[iy];
            let y1 = self.data_y[iy + 1];
            let z00 = (self.func_f)(x0, y0) - self.thr;
            let z10 = (self.func_f)(x1, y0) - self.thr;
            let z01 = (self.func_f)(x0, y1) - self.thr;
            let z11 = (self.func_f)(x1, y1) - self.thr;
            let b00 = z00 < v0;
            let b10 = z10 < v0;
            let b01 = z01 < v0;
            let b11 = z11 < v0;
            if b00 != b10 && b00 == b01 && b00 == b11 {
                let xx0 = bilinear(x0, x1, z00, z10);
                let yy1 = bilinear(y0, y1, z10, z11);
                Ok(((xx0, y0), (x1, yy1)))
            }
            else if b00 == b10 && b00 != b01 && b00 == b11 {
                let yy0 = bilinear(y0, y1, z00, z01);
                let xx1 = bilinear(x0, x1, z01, z11);
                Ok(((x0, yy0), (xx1, y1)))
            }
            else if b00 == b10 && b00 == b01 && b00 != b11 {
                let xx0 = bilinear(x0, x1, z01, z11);
                let yy1 = bilinear(y0, y1, z10, z11);
                Ok(((xx0, y1), (x1, yy1)))
            }
            else if b00 != b10 && b00 == b01 && b00 != b11 {
                let xx0 = bilinear(x0, x1, z00, z10);
                let xx1 = bilinear(x0, x1, z01, z11);
                Ok(((xx0, y0), (xx1, y1)))
            }
            else if b00 == b10 && b00 != b01 && b00 != b11 {
                let yy0 = bilinear(y0, y1, z00, z01);
                let yy1 = bilinear(y0, y1, z10, z11);
                Ok(((x0, yy0), (x1, yy1)))
            }
            else if b00 != b10 && b00 != b01 && b00 == b11 {
                Err(true)
            }
            else if b00 != b10 && b00 != b01 && b00 != b11 {
                let xx1 = bilinear(x0, x1, z00, z10);
                let yy1 = bilinear(y0, y1, z00, z01);
                Ok(((x0, yy1), (xx1, y0)))
            }
            else {
                Err(true)
            }
        }
        else {
            Err(false)
        }
    }
}

impl<DB: DrawingBackend, V: 'static + Clone, F> Iterator for ContourSeries<DB, V, F>
where V: Float, F: Fn(V, V) -> V
{
    type Item = DynElement<'static, DB, (V, V)>;

    fn next(&mut self) -> Option<Self::Item>
    {
        loop {
            match self.next_line() {
                Err(true) => continue,
                Err(false) => return None,
                Ok(((x0, y0), (x1, y1))) => {
                    return Some(PathElement::new(vec![(x0, y0), (x1, y1)], self.style.clone()).into_dyn())
                }
            }
        }
    }
}
