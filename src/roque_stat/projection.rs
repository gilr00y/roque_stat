use std::borrow::BorrowMut;
use std::collections::HashMap;
use ndarray::{Array1, Array2};
use rand::distributions::uniform::SampleBorrow;
use crate::roque_stat::batch_crp::BatchCRP;

pub struct Projection {
  pub crp: BatchCRP
}

impl Projection {
  pub fn new(crp: &BatchCRP, projected_dims: Vec<usize>) -> Projection {
    // NOTE: Projected dims are the indexes, not a mask.
    // let n_dims = crp.psi_scale.len();
    // let mut A: Array2<f64> = Array2::zeros((n_dims, n_dims));//from_diag(Array);
    // let mut a: Array1<f64> = Array1::zeros(n_dims);//from_diag(Array);

    // for dim_idx in projected_dims {
    //   A[[dim_idx, dim_idx]] = 1.0;
    //   a[dim_idx] = 1.0;
    // }

    let new_tables =crp.tables.iter().map(|entry| {
      match entry {
        (tbl_id, tbl) => {
          let mut new_tbl = (*(*tbl)).clone();
          let mut new_psi: Array2<f64> = Array2::zeros((projected_dims.len(), projected_dims.len()));
          let mut new_mu: Array1<f64> = Array1::zeros(projected_dims.len());

          for dim1 in &projected_dims {
            let dim_a = dim1.clone();
            new_mu[dim_a] = tbl.component.mu[dim_a];
            for dim2 in &projected_dims {
              let dim_b = dim2.clone();
              new_psi[[dim_a, dim_b]] = tbl.component.psi[[dim_a, dim_b]];
            }
          }
          new_tbl.component.mu = new_mu;//(new_tbl.component.mu * &A * &a).remove_axis(ndarray::Axis(1));
          new_tbl.component.psi = new_psi;//&A * new_tbl.component.psi * &A.t();
          new_tbl.component.n_dim = projected_dims.len();
          (tbl_id.clone(), Box::new(new_tbl))
        }
      }
    });

    Projection {
      crp: BatchCRP {
        alpha: crp.alpha,
        max_iterations: crp.max_iterations,
        tables:  Box::new(HashMap::from_iter(new_tables)),
        psi_scale: crp.psi_scale.clone()
      }
    }
  }
}
