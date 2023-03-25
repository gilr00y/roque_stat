// use ndarray::{Array1, Array2};
// use ndarray_linalg::cholesky::Cholesky;
// use std::f64::consts::LN_2_PI;
//
// #[derive(Debug, Clone)]
// struct Component {
//   mu: Array1<f64>,
//   psi: Array2<f64>,
//   weight: f64,
// }
//
// fn kl_divergence_gaussians(c1: &Component, c2: &Component) -> f64 {
//   let n = c1.mu.len();
//   let inv_psi2 = c2.psi.inv().unwrap();
//   let psi1_sqrt_det = Cholesky::cholesky(c1.psi.clone()).unwrap().det().sqrt();
//   let psi2_sqrt_det = Cholesky::cholesky(c2.psi.clone()).unwrap().det().sqrt();
//
//   let delta_mu = &c1.mu - &c2.mu;
//
//   0.5 * (inv_psi2.dot(&c1.psi).trace()
//     - n as f64
//     - 2.0 * (psi1_sqrt_det / psi2_sqrt_det).ln()
//     + delta_mu.t().dot(&inv_psi2).dot(&delta_mu))
// }
//
// fn kl_divergence_bounds(gmm1: &[Component], gmm2: &[Component]) -> (f64, f64) {
//   let mut lower_bound = 0.0;
//   let mut upper_bound = 0.0;
//
//   for c1 in gmm1 {
//     let mut min_kl = f64::INFINITY;
//     let mut max_kl = f64::NEG_INFINITY;
//
//     for c2 in gmm2 {
//       let kl = kl_divergence_gaussians(c1, c2);
//       min_kl = min_kl.min(kl);
//       max_kl = max_kl.max(kl);
//     }
//
//     lower_bound += c1.weight * min_kl;
//     upper_bound += c1.weight * max_kl;
//   }
//
//   (lower_bound, upper_bound)
// }
