use ndarray::{Array1, Array2};
use ndarray_linalg::cholesky::Cholesky;

use ndarray_linalg::{Determinant, Inverse, UPLO, Trace};
use ndarray_rand::rand_distr::num_traits::real::Real;
use crate::roque_stat::batch_crp::BatchCRP;
use crate::roque_stat::crp::CRP;
use crate::roque_stat::table::Table;

pub struct KLDivergence {
  pub upper: f64,
  pub lower: f64,
  pub mean: f64
}

impl KLDivergence {
  pub fn new(crp1: &BatchCRP, crp2: &BatchCRP) -> Self {
    let (lower, upper) = kl_divergence_bounds(crp1, crp2);
    KLDivergence {
      upper,
      lower,
      mean: (upper + lower)/2.0
    }
  }
}
fn kl_divergence_gaussians(t1: &Table, t2: &Table) -> f64 {
  let n = t1.component.mu.len();
  let inv_psi2 = t2.component.psi.inv().unwrap();
  let psi1_sqrt_det = t1.component.psi.cholesky(UPLO::Lower).unwrap()
    .det().unwrap()
    .sqrt();
  let psi2_sqrt_det = t2.component.psi.cholesky(UPLO::Lower).unwrap()
    .det().unwrap()
    .sqrt();

  let delta_mu = &t1.component.mu - &t2.component.mu;

  0.5 * (inv_psi2.dot(&t1.component.psi).trace().unwrap()
    - n as f64
    - 2.0 * (psi1_sqrt_det / psi2_sqrt_det).ln()
    + delta_mu.t().dot(&inv_psi2).dot(&delta_mu))
}

fn kl_divergence_bounds(crp1: &BatchCRP, crp2: &BatchCRP) -> (f64, f64) {
  let mut lower_bound = 0.0;
  let mut upper_bound = 0.0;

  for t1 in crp1.tables.values() {
    let mut min_kl = f64::INFINITY;
    let mut max_kl = f64::NEG_INFINITY;

    for t2 in crp2.tables.values() {
      let kl = kl_divergence_gaussians(t1, t2);
      min_kl = min_kl.min(kl);
      max_kl = max_kl.max(kl);
    }

    lower_bound += t1.count as f64 * min_kl;
    upper_bound += t1.count as f64 * max_kl;
  }

  (lower_bound, upper_bound)
}
