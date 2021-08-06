use totsu::prelude::*;
use totsu::operator::MatBuild;
use totsu::linalg::F64LAPACK;
use totsu::problem::ProbSOCP;

use std::collections::HashMap;
use std::io::prelude::*;
use std::io::BufWriter;
use std::fs::File;

use itertools::iproduct;

use intel_mkl_src as _;

type AMatBuild = MatBuild<F64LAPACK, f64>;
type AProbSOCP = ProbSOCP<F64LAPACK, f64>;
type ASolver = Solver<F64LAPACK, f64>;

struct Member
{
    e: f64,
    head_idx: usize,
    tail_idx: usize,
}

struct Node
{
    r: (f64, f64),
    p: (Option<f64>, Option<f64>),
    dof_idx: Option<usize>,
}

impl Member
{
    fn v(&self, nodes: &Vec<Node>) -> f64
    {
        let head = &nodes[self.head_idx];
        let tail = &nodes[self.tail_idx];
        (head.r.0 - tail.r.0).hypot(head.r.1 - tail.r.1)
    }
}

fn make_nodes(x_num: u16, y_num: u16) -> (Vec<Node>, HashMap<(u16, u16), usize>)
{
    let mut nodes = Vec::<Node>::new();
    let mut nodeidx = HashMap::<(u16, u16), usize>::new();

    for (x, y) in iproduct!(0..x_num, 0..y_num) {
        let idx = nodes.len();
        nodes.push(Node {
            r: (x as f64, y as f64),
            p: (Some(0.), Some(0.)),
            dof_idx: None,
        });
        let r = nodeidx.insert((x, y), idx);
        assert_eq!(r, None);
    }

    (nodes, nodeidx)
}

fn make_torus(nodeidx: &HashMap<(u16, u16), usize>, x_num: u16, y_num: u16) -> Vec<Member>
{
    let mut members = Vec::<Member>::new();

    for (hx, hy) in iproduct!(0..x_num, 0..y_num) {
        //println!("{},{}", hx, hy);

        if let Some(&head_idx) = nodeidx.get(&(hx, hy)) {
            //println!("  {}", head_idx);

            for (dx, dy) in [(1, 0), (0, 1), (1, 1)] {
                let tx = hx + dx;
                let ty = hy + dy;

                //println!("  {},{}", tx, ty);

                if let Some(&tail_idx) = nodeidx.get(&(tx, ty)) {
                    //println!("    {}", tail_idx);

                    //println!("      {} {}", head_idx, tail_idx);
                    members.push(Member {
                        e: 1.0,
                        head_idx,
                        tail_idx,
                    });
                }
            }
        }
    }

    members
}

fn nodes_dof(nodes: &mut Vec<Node>) -> usize
{
    let mut dof = 0;
    for n in nodes {
        n.dof_idx = Some(dof);
        if n.p.0.is_some() {dof += 1;}
        if n.p.1.is_some() {dof += 1;}
    }
    dof
}

/// main
fn main() -> std::io::Result<()> {
    env_logger::init();

    //----- make members and nodes

    let x_num: u16 = 3;
    let y_num: u16 = 3;

    let (mut nodes, nodeidx) = make_nodes(x_num, y_num);
    let members = make_torus(&nodeidx, x_num, y_num);

    // fixed DOF of nodes
    for y in 0..y_num {
        let node = &mut nodes[nodeidx[&(0, y)]];
        node.p = (None, None)
    }
    // external force on nodes
    {
        let node = &mut nodes[nodeidx[&(x_num - 1, y_num - 1)]];
        node.p = (Some(1.), Some(-1.));
    }

    let l = members.len();
    let vol_ratio = 0.5;

    //----- formulate compliance minimization as SOCP

    let n = l * 3/*x,q,w*/;
    let m = l * 2 + 1;
    let p = nodes_dof(&mut nodes);

    let vec_f = AMatBuild::new(MatType::General(n, 1))
                .by_fn(|r, _| if r < l * 2 {0./*x,q*/} else {2./*w*/});
    //println!("{}", vec_f);

    let mut mats_g = Vec::<AMatBuild>::new();
    for i in 0..l {
        let mut mat_g = AMatBuild::new(MatType::General(2, n));
        mat_g[(0, i/*x_i*/)] = -1.;
        mat_g[(0, l * 2 + i/*w_i*/)] = 1.;
        mat_g[(1, l + i/*q_i*/)] = (2. * members[i].v(&nodes) / members[i].e).sqrt();
        mats_g.push(mat_g);
    }
    for _ in l..m {
        mats_g.push(AMatBuild::new(MatType::General(0, n)));
    }
    //println!("{}", mats_g.len());

    let mut vecs_h = Vec::<AMatBuild>::new();
    for _ in 0..l {
        vecs_h.push(AMatBuild::new(MatType::General(2, 1)));
    }
    for _ in l..m {
        vecs_h.push(AMatBuild::new(MatType::General(0, 1)));
    }
    //println!("{}", vecs_h.len());

    let mut vecs_c = Vec::<AMatBuild>::new();
    for i in 0..l {
        let mut vec_c = AMatBuild::new(MatType::General(n, 1));
        vec_c[(i/*x_i*/, 0)] = 1.;
        vec_c[(l * 2 + i/*w_i*/, 0)] = 1.;
        vecs_c.push(vec_c);
    }
    for i in 0..l {
        let mut vec_c = AMatBuild::new(MatType::General(n, 1));
        vec_c[(i/*x_i*/, 0)] = 1.;
        vecs_c.push(vec_c);
    }
    {
        let vec_c = AMatBuild::new(MatType::General(n, 1))
                        .by_fn(|r, _| if r < l {-members[r].v(&nodes)} else {0.});
        vecs_c.push(vec_c);
    }
    //println!("{}", vecs_c.len());

    let mut scls_d = vec![0.; m];
    scls_d[m - 1] = members.iter().fold(0., |acc, member| acc + member.v(&nodes)) * vol_ratio;
    //println!("{:?}", scls_d);

    let mut mat_a = AMatBuild::new(MatType::General(p, n));
    for (i, member) in members.iter().enumerate() {
        let head = &nodes[member.head_idx];
        let tail = &nodes[member.tail_idx];
        let beta = (tail.r.0 - head.r.0, tail.r.1 - head.r.1);
        let norm = beta.0.hypot(beta.1);
        let beta = (beta.0 / norm, beta.1 / norm);

        let mut dof_idx = head.dof_idx.unwrap();
        if head.p.0.is_some() {
            mat_a[(dof_idx, l + i/*q_i*/)] += -beta.0;
            dof_idx += 1;
        }
        if head.p.1.is_some() {
            mat_a[(dof_idx, l + i/*q_i*/)] += -beta.1;
        }
        let mut dof_idx = tail.dof_idx.unwrap();
        if tail.p.0.is_some() {
            mat_a[(dof_idx, l + i/*q_i*/)] += beta.0;
            dof_idx += 1;
        }
        if tail.p.1.is_some() {
            mat_a[(dof_idx, l + i/*q_i*/)] += beta.1;
        }
    }
    //println!("{}", mat_a);

    let mut vec_b = AMatBuild::new(MatType::General(p, 1));
    for node in nodes.iter() {
        let mut dof_idx = node.dof_idx.unwrap();
        if let Some(p0) = node.p.0 {
            vec_b[(dof_idx, 0)] = p0;
            dof_idx += 1;
        }
        if let Some(p1) = node.p.1 {
            vec_b[(dof_idx, 0)] = p1;
        }
    }
    //println!("{}", vec_b);

    //----- solve SOCP

    let s = ASolver::new().par(|p| {
        p.eps_acc = 1e-3;
    });
    let mut socp = AProbSOCP::new(vec_f, mats_g, vecs_h, vecs_c, scls_d, mat_a, vec_b);
    let rslt = s.solve(socp.problem()).unwrap();
    //println!("{:?}", rslt);

/*
    //----- file output for graph plot
    // TODO: use some plotting crate

    let mut dat_point = BufWriter::new(File::create("dat_point")?);

    for smp in 0 .. l {
        writeln!(dat_point, "{} {} {} {}", x[(0, smp)], x[(1, smp)], y[(smp, 0)], alpha[smp])?;
    }

    let mut dat_grid = BufWriter::new(File::create("dat_grid")?);

    let grid = 20;
    for iy in 0 .. grid {
        for ix in 0 .. grid {
            let xi = AMatBuild::new(MatType::General(2, 1))
                     .iter_colmaj(&[
                         ix as f64 / (grid - 1) as f64,
                         iy as f64 / (grid - 1) as f64
                     ]);

            let f = wx(&x, &y, alpha, &xi, 0) + bias;

            write!(dat_grid, " {}", f)?;
        }
        writeln!(dat_grid)?;
    }
*/

    Ok(())
}
