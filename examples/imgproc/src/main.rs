use totsu::prelude::*;
use totsu::predef::SOCP;
use totsu::mat::XOR64;

#[macro_use] extern crate itertools;

fn clamp(v: FP) -> u8 {
    let v = if v < 0. {0.} else {v};
    let v = if v > 1. {1.} else {v};
    (v * 255.) as u8
}

/// main
fn main() -> std::io::Result<()> {
    let mut rnd = XOR64::init();

    let sz = 8;
    let ch = 3;

    let n: usize = sz * sz * ch + (sz - 2) * (sz - 2);
    let m: usize = (sz - 2) * (sz - 2);
    let p: usize = (sz * 4 - 4) * ch;

    let mut vec_f = Mat::new_vec(n);
    vec_f.rows_mut(sz * sz * ch .. n).assign_all(1.);

    let mut mat_g = vec![Mat::new(ch, n); m];
    let vec_h = vec![Mat::new_vec(ch); m];
    let mut vec_c = vec![Mat::new_vec(n); m];
    let scl_d = vec![0. as FP; m];
    for (x, y) in iproduct!(0 .. sz - 2, 0 .. sz - 2) {
        let i = x + y * (sz - 2);
        for c in 0 .. ch {
            let co = c * sz * sz;
            mat_g[i][(c, (x + 0) + (y + 1) * sz + co)] = 2.;
            mat_g[i][(c, (x + 2) + (y + 1) * sz + co)] = 2.;
            mat_g[i][(c, (x + 1) + (y + 0) * sz + co)] = 2.;
            mat_g[i][(c, (x + 1) + (y + 2) * sz + co)] = 2.;
            mat_g[i][(c, (x + 1) + (y + 1) * sz + co)] = -4.;
            mat_g[i][(c, (x + 0) + (y + 0) * sz + co)] = -1.;
            mat_g[i][(c, (x + 2) + (y + 0) * sz + co)] = -1.;
            mat_g[i][(c, (x + 0) + (y + 2) * sz + co)] = -1.;
            mat_g[i][(c, (x + 2) + (y + 2) * sz + co)] = -1.;
        }
        vec_c[i][(sz * sz * ch + i, 0)] = 1.;
    }

    let mut mat_a = Mat::new(p, n);
    let vec_b = Mat::new_vec(p).set_by(|_, _| rnd.next() * 0.5 + 0.25);
    let mut ip = 0;
    for c in 0 .. ch {
        let co = c * sz * sz;
        for x in 0 .. sz {
            mat_a[(ip, x + co)] = 1.;
            ip += 1;
            mat_a[(ip, x + (sz - 1) * sz + co)] = 1.;
            ip += 1;
        }
        for y in 1 .. sz - 1 {
            mat_a[(ip, y * sz + co)] = 1.;
            ip += 1;
            mat_a[(ip, (sz - 1) + y * sz + co)] = 1.;
            ip += 1;
        }
    }

    //let param = PDIPMParam::default();
    let param = PDIPMParam {
        eps: 1e-3,
        .. Default::default()
    };
    let rslt = PDIPM::new().solve_socp(&param, &mut std::io::sink(),
                                       &vec_f,
                                       &mat_g, &vec_h, &vec_c, &scl_d,
                                       &mat_a, &vec_b).unwrap();

    let mut img = image::ImageBuffer::new(sz as u32, sz as u32);
    for (x, y, p) in img.enumerate_pixels_mut() {
        let xx = x as usize;
        let yy = y as usize;
        let r = clamp(rslt[(xx + yy * sz, 0)]);
        let g = clamp(rslt[(xx + yy * sz + sz * sz, 0)]);
        let b = clamp(rslt[(xx + yy * sz + 2 * sz * sz, 0)]);
        *p = image::Rgb([r, g, b]);
    }
    img.save("img.bmp")?;

    Ok(())
}
