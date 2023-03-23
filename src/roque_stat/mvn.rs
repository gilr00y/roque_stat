use ndarray::{Array, Array1, Array2, Axis, Ix2, Shape, Zip};
use ndarray_rand::{rand_distr::StandardNormal, RandomExt};
use ndarray_linalg::{Cholesky, Determinant, Inverse, UPLO};
use rand::Rng;
use std::f64::consts::PI;
use crate::roque_stat::multivariate_t::MultivariateT;
use crate::roque_stat::niw::NormalInverseWishart;
use crate::roque_stat::util;

// Define a struct for multivariate normal distribution
#[derive(Clone, Debug)]
pub struct MVN {
  pub(crate) n_dim: usize,
  pub(crate) mu: Array1<f64>, // Mean
  pub(crate) psi: Array2<f64>, // Covariance
  pub(crate) k0: i32, // Scale factor
}

impl MVN {
  // Constructor for the MVN struct
  fn new(n_dim: usize) -> Self {
    let mu = Array1::zeros(n_dim); // Initialize mean vector with zeros
    // Initialize covariance matrix with random values from standard normal distribution
    let psi: Array2<f64> = Array::random((n_dim, n_dim), StandardNormal);
    let k0 = 1; // Initialize scale factor
    MVN { n_dim, mu, psi, k0 } // Return the MVN struct
  }

  fn niw(&self) -> NormalInverseWishart {
    NormalInverseWishart {
      dist_mean: self.mu.clone(),
      k0: self.k0,
      v0: self.k0,
      psi: self.psi.clone()
    }
  }

  // Function to draw random samples from the MVN distribution
  pub(crate) fn draw(&self, n: usize, compress: bool) -> Array1<f64> {
    // Check if the scale factor is less than or equal to 5 and compress flag is true
    if self.k0 <= 5 && compress {
      // Return mean vector as a row vector
      self.mu.clone()
    } else {
      // Calculate scaled covariance matrix &
      // (TODO: Truncate the values in the scaled covariance matrix to two decimal places)
      let scaled_sig = &self.sig();
      // println!("SCALED SIG: {scaled_sig}");
      // Ensure that the diagonal elements of the scaled covariance matrix are greater than a threshold value
      let clipped_scaled_sig = util::ensure_diag_above_threshold(&scaled_sig, 0.1f64, self.n_dim);
      // println!("CLIPPED SCALED SIG: {clipped_scaled_sig}");
      // Clone the mean vector and generate a random vector from standard normal distribution
      let scaled_mu = self.mu.clone();
      let z: Array1<f64> = Array::random(self.n_dim, StandardNormal);
      // println!("Z: {z}");
      // Compute the Cholesky decomposition of the clipped scaled covariance matrix
      // let j_scaled_sig = Cholesky::new(clipped_scaled_sig.clone().).unwrap().l();
      let j_scaled_sig = clipped_scaled_sig.cholesky(UPLO::Lower).unwrap();
      // println!("J SCALED SIG: {j_scaled_sig}");
      let root = Array2::from_shape_vec((self.n_dim, self.n_dim), j_scaled_sig.as_slice().unwrap().to_vec()).unwrap();
      // println!("ROOT: {root}");
      // Compute the random samples from the MVN distribution
      // 2x2 * 1x2
      let result = &z.dot(&root) + &scaled_mu;//.insert_axis(Axis(0));
      // println!("RESULT: {result}");
      // Check if the random samples have any values greater than 1e8
      if result.iter().any(|&x| x > 1e8) {
        println!("Absurd sample received: MU: {:?}, SIG: {:?}", scaled_mu, clipped_scaled_sig);
      }
      result
    }
  }

  // Function to compute the scaled covariance matrix
  fn sig(&self) -> Array2<f64> {
    // println!("SIG K0: {}", self.k0);
    // println!("SIG PSI: {}", self.psi);
    let adj = if self.k0 + 1 <= self.n_dim as i32 { 2 } else {
      (self.k0 * (self.k0 - self.n_dim as i32 + 1))
    };
    ((self.k0 + 1) as f64) / (self.k0 as f64 * adj as f64) * &self.psi
  }

  // Function to update the MVN distribution with new data
  pub(crate) fn update(&mut self, new_data: &Array2<f64>) {
    let new_niw = self.niw().update(new_data);
    self.mu = new_niw.dist_mean;
    self.psi = new_niw.psi;
    self.k0 = new_niw.k0;
    // MVN {
    //   n_dim: self.n_dim,
    //   mu: new_niw.dist_mean,
    //   psi: new_niw.psi,
    //   k0: new_niw.k0,
    // }
  }

  fn remove(&self, data: &Array2<f64>) -> Self {
    self.clone()
    // let new_niw = NormalInverseWishart::remove(&self.mu, self.k0, self.k0, &self.psi, data);
    // MVN {
    //   n_dim: self.n_dim,
    //   mu: new_niw.dist_mean,
    //   psi: new_niw.psi,
    //   k0: new_niw.k0,
    // }
  }

  pub(crate) fn pp(&self, x: &Array1<f64>) -> f64 {
    let mvt = MultivariateT::new(self.mu.clone(), self.psi.clone(), (self.n_dim + 2) as u32);
    mvt.posterior_predictive(x, &self.sig())
  }
}
