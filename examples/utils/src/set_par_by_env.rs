use totsu::solver::SolverParam;
use num_traits::Float;
use num_traits::Num;

pub fn num_by_env<N: Num + std::fmt::Display>(e: &str) -> Option<N>
{
    if let Some(v) = std::env::var(e).ok()
                     .and_then(|s| {N::from_str_radix(&s, 10).ok()}) {
        log::info!("{}: {}", e, v);
        Some(v)
    }
    else {
        None
    }
}

pub fn set_par_by_env<F: Float + std::fmt::Display>(p: &mut SolverParam<F>)
{
    p.max_iter = num_by_env("MAX_ITER").or(p.max_iter);
    p.eps_acc = num_by_env("EPS_ACC").unwrap_or(p.eps_acc);
    p.eps_inf = num_by_env("EPS_INF").unwrap_or(p.eps_inf);
    p.eps_zero = num_by_env("EPS_ZERO").unwrap_or(p.eps_zero);
    p.log_period = num_by_env("LOG_PERIOD").unwrap_or(p.log_period);
}
