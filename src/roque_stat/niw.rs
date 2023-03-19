use ndarray::{Array1, Array2};
// use ndarray_stats::QuantileExt;
use std::cmp;
use crate::roque_stat::util;

pub struct NormalInverseWishart {
  pub dist_mean: Array1<f64>,
  pub k0: i32,
  pub v0: i32,
  pub psi: Array2<f64>,
}

impl NormalInverseWishart {
  fn remove(&self, samples: Array2<f64>) -> NormalInverseWishart {
    let num_samples = samples.nrows() as f64;
    let n_dim = self.dist_mean.len();

    let sample_mean = samples.mean_axis(ndarray::Axis(0)).unwrap();
    let sample_mean_broadcast = sample_mean.broadcast((num_samples as usize, n_dim)).unwrap();
    let new_mean = (self.k0 as f64 * &self.dist_mean - num_samples * &sample_mean)
      / (self.k0 as f64 - num_samples);

    let k1 = self.k0 - num_samples as i32;
    let v1 = self.v0 - num_samples as i32;

    let deviations = samples - &sample_mean_broadcast;
    let c = deviations.t().dot(&deviations);

    let mean_diff = &sample_mean - &new_mean;
    let mean_adj_coeff = (num_samples * k1 as f64) / (k1 as f64 + num_samples);
    let mean_psi_adjustment = mean_diff.dot(&mean_diff.t());

    let base_psi = &self.psi - &c - mean_adj_coeff * mean_psi_adjustment;
    let clipped_psi = util::ensure_diag_above_threshold(&base_psi, 0.1, n_dim);
    // let clipped_psi = Util::ensure_diag_above_threshold(base_psi, 0.1, n_dim);

    NormalInverseWishart {
      dist_mean: new_mean,
      k0: k1,
      v0: v1,
      psi: clipped_psi.clone(),
    }
  }

  pub(crate) fn update(&self, samples: &Array2<f64>) -> NormalInverseWishart {
    let num_samples = samples.nrows() as f64;
    let n_dim = self.dist_mean.len();

    let sample_mean = samples.mean_axis(ndarray::Axis(0)).unwrap();
    let sample_mean_broadcast = sample_mean.broadcast((num_samples as usize, n_dim)).unwrap();
    let new_mean = (self.k0 as f64 * &self.dist_mean + num_samples * &sample_mean)
      / (self.k0 as f64 + num_samples);

    let k1 = self.k0 + num_samples as i32;
    let v1 = self.v0 + num_samples as i32;

    let deviations = samples - &sample_mean_broadcast;
    let c = deviations.t().dot(&deviations);

    let mean_diff = &sample_mean - &self.dist_mean;
    let mean_adj_coeff = (num_samples * self.k0 as f64) / (self.k0 as f64 + num_samples);
    let mean_psi_adjustment = mean_diff.dot(&mean_diff.t());

    let new_psi = &self.psi + &c + mean_adj_coeff * mean_psi_adjustment;

    NormalInverseWishart {
      dist_mean: new_mean,
      k0: k1,
      v0: v1,
      psi: new_psi,
    }
  }
}

