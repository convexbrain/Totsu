use totsu::solver::SolverParam;
use num_traits::Float;
use num_traits::Num;

pub fn set_par_by_env<F: Float>(p: &mut SolverParam<F>)
{
    if let Some(v) = std::env::var("MAX_ITER").ok()
                     .and_then(|s| {s.parse().ok()}) {
        p.max_iter = Some(v);
    }
    if let Some(v) = std::env::var("EPS_ACC").ok()
                     .and_then(|s| {<F as Num>::from_str_radix(&s, 10).ok()}) {
        p.eps_acc = v;
    }
    if let Some(v) = std::env::var("EPS_INF").ok()
                     .and_then(|s| {<F as Num>::from_str_radix(&s, 10).ok()}) {
        p.eps_inf = v;
    }
    if let Some(v) = std::env::var("EPS_ZERO").ok()
                     .and_then(|s| {<F as Num>::from_str_radix(&s, 10).ok()}) {
        p.eps_zero = v;
    }
    if let Some(v) = std::env::var("LOG_PERIOD").ok()
                     .and_then(|s| {s.parse().ok()}) {
        p.log_period = v;
    }
}
