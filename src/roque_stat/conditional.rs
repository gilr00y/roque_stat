use std::collections::HashMap;
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::{Inverse, Determinant};
use ndarray_rand::rand_distr::num_traits::ToPrimitive;
use rand::distributions::uniform::SampleBorrow;
use crate::roque_stat::batch_crp::BatchCRP;

pub struct Conditional {
  pub crp: BatchCRP,
}

fn multivariate_normal_pdf(x: &Array1<f64>, mu: &Array1<f64>, psi: &Array2<f64>) -> f64 {
  let n = x.len();
  let sqrt_det_psi = psi.det().unwrap().sqrt();
  let psi_inv = psi.inv().unwrap();
  let diff = x - mu;

  let exponent_term = -0.5 * diff.t().dot(&psi_inv).dot(&diff);
  let normalization_term = ((2.0 * std::f64::consts::PI).powi(n as i32) * sqrt_det_psi).sqrt();

  let pdf_value = (1.0 / normalization_term) * f64::exp(exponent_term);

  pdf_value
}

pub type ConditionalQuery = HashMap<usize, f64>;

impl Conditional {

  pub fn new(crp: &BatchCRP, query: ConditionalQuery) -> Self {
    // Set up partitions 1 & 2
    // TODO: Set an explicit "n_dims" prop on crp.
    let n_dims_total = crp.psi_scale.len();
    let x2_size = query.len();
    let x1_size = n_dims_total - x2_size;

    // let idx_22: Array1<usize> = Array1::from_iter(query.keys().iter());
    // let idx_11_iter = (0..n_dims_total)
    //   .into_iter()
    //   .filter(|x|!idx_22.contains(x));
    // let idx_11 = Array1::from_iter(idx_11_iter);

    // let idx_22: Vec<usize> = query.keys().cloned().collect();
    // let mut idx_11: Vec<usize> = (0..n).collect();
    // idx_11.retain(|x| !idx_22.contains(x));

    let query_values: Array1<f64> = Array1::from(query.values().cloned().collect::<Vec<_>>());

    let idx_22: Vec<usize> = query.keys().cloned().collect();
    let mut idx_11: Vec<usize> = (0..n_dims_total).collect();
    idx_11.retain(|x| !idx_22.contains(x));

    let total_likelihood: f64 = crp.tables.iter().map(|entry| {
      match entry {
        (_tbl_id, tbl) => {
          let mu2 = tbl.component.mu.select(Axis(0), &idx_22);
          let psi22 = tbl.component.psi.select(Axis(0), &idx_22).select(Axis(1), &idx_22);
          let likelihood = multivariate_normal_pdf(&query_values, &mu2, &psi22);
          likelihood
        }
      }
    }).sum();

    let new_tables = crp.tables.iter().map(|entry| {
      match entry {
        (tbl_id, tbl) => {
          let mut new_tbl = (*(*tbl)).clone();

          let mu1 = tbl.component.mu.select(Axis(0), &idx_11);
          let mu2 = tbl.component.mu.select(Axis(0), &idx_22);

          let psi11 = tbl.component.psi.select(Axis(0), &idx_11).select(Axis(1), &idx_11);
          let psi12 = tbl.component.psi.select(Axis(0), &idx_11).select(Axis(1), &idx_22);
          let psi21 = tbl.component.psi.select(Axis(0), &idx_22).select(Axis(1), &idx_11);
          let psi22 = tbl.component.psi.select(Axis(0), &idx_22).select(Axis(1), &idx_22);

          let x2: Array1<f64> = Array1::from(query.values().cloned().collect::<Vec<_>>());
          let mu_cond = &mu1 + psi12.dot(&psi22.inv().unwrap()).dot(&(&x2 - &mu2));
          let psi_cond = psi11 - psi12.dot(&psi22.inv().unwrap()).dot(&psi21);

          let likelihood = multivariate_normal_pdf(&query_values, &mu2, &psi22);

          new_tbl.component.mu = mu_cond;//(new_tbl.component.mu * &A * &a).remove_axis(ndarray::Axis(1));
          new_tbl.component.psi = psi_cond;//&A * new_tbl.component.psi * &A.t();
          new_tbl.component.n_dim = idx_11.len();
          new_tbl.count = (tbl.count as f64 * likelihood /total_likelihood.clone()).to_u64().unwrap();

          (tbl_id.clone(), Box::new(new_tbl))
        }
      }
    });

    // Clone each table and set the new mu/psi.
    Conditional {
      crp: BatchCRP {
        alpha: crp.alpha,
        max_iterations: crp.max_iterations,
        tables: Box::new(HashMap::from_iter(new_tables)),
        psi_scale: crp.psi_scale.clone().select(Axis(0), &idx_11),
      }
    }
  }
}
