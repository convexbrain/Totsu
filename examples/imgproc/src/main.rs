use totsu::prelude::*;
use totsu::predef::SOCP;
use totsu::mat::XOR64;

#[macro_use] extern crate itertools;

/// main
fn main() -> std::io::Result<()> {
    let mut rnd = XOR64::init();

    let sz = 8;

    let alpha = 0.5;
    let beta = 1.;

    let n: usize = sz * sz + (sz - 2) * (sz - 2) + 1;
    let m: usize = (sz - 2) * (sz - 2) + 1;
    let p: usize = 0;

    let mut vec_f = Mat::new_vec(n);
    vec_f.rows_mut(sz * sz .. sz * sz + (sz - 2) * (sz - 2)).assign_all(alpha);
    vec_f[(n - 1, 0)] = beta;

    let mut mat_g = vec![Mat::new(0, 0); m];
    let mut vec_h = vec![Mat::new_vec(0); m];
    let mut vec_c = vec![Mat::new_vec(n); m];
    let scl_d = vec![0. as FP; m];
    for (x, y) in iproduct!(0 .. sz - 2, 0 .. sz - 2) {
        let i = x + y * (sz - 2);
        mat_g[i] = Mat::new(1, n);
        vec_h[i] = Mat::new_vec(1);
        mat_g[i][(0, (x + 0) + (y + 1) * sz)] = 2.;
        mat_g[i][(0, (x + 2) + (y + 1) * sz)] = 2.;
        mat_g[i][(0, (x + 1) + (y + 0) * sz)] = 2.;
        mat_g[i][(0, (x + 1) + (y + 2) * sz)] = 2.;
        mat_g[i][(0, (x + 1) + (y + 1) * sz)] = -4.;
        mat_g[i][(0, (x + 0) + (y + 0) * sz)] = -1.;
        mat_g[i][(0, (x + 2) + (y + 0) * sz)] = -1.;
        mat_g[i][(0, (x + 0) + (y + 2) * sz)] = -1.;
        mat_g[i][(0, (x + 2) + (y + 2) * sz)] = -1.;
        vec_c[i][(sz * sz + i, 0)] = 1.;
    }
    mat_g[m - 1] = Mat::new(sz * sz, n).set_eye(-1.);
    vec_h[m - 1] = Mat::new_vec(sz * sz).set_by(|_, _| rnd.next());
    vec_c[m - 1][(n - 1, 0)] = 1.;
    
    let mat_a = Mat::new(p, n);
    let vec_b = Mat::new_vec(p);

    let mut img = image::ImageBuffer::new(sz as u32, sz as u32);
    for (x, y, p) in img.enumerate_pixels_mut() {
        let v = vec_h[m - 1][(x as usize + y as usize * sz, 0)];
        let g = (v * 255.) as u8;
        *p = image::Rgb([g, g, g]);
    }
    img.save("img.bmp")?;

    //let param = PDIPMParam::default();
    let param = PDIPMParam {
        eps: 1e-3,
        .. Default::default()
    };
    let rslt = PDIPM::new().solve_socp(&param, &mut std::io::sink(),
                                       &vec_f,
                                       &mat_g, &vec_h, &vec_c, &scl_d,
                                       &mat_a, &vec_b).unwrap();

    for (x, y, p) in img.enumerate_pixels_mut() {
        let v = rslt[(x as usize + y as usize * sz, 0)];
        let v = if v < 0. {0.} else {v};
        let v = if v > 1. {1.} else {v};
        let g = (v * 255.) as u8;
        *p = image::Rgb([g, g, g]);
    }
    img.save("img_o.bmp")?;

    Ok(())
}
