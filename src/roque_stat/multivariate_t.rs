use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2, NdFloat};
use ndarray_linalg::{Determinant, Solve};

#[derive(Debug)]
pub struct MultivariateT {
  mu: Array1<f64>,
  psi: Array2<f64>,
  dof: u32,
  n_dim: u32,
}

impl MultivariateT {
  pub(crate) fn new(mu: Array1<f64>, psi: Array2<f64>, dof: u32) -> MultivariateT {
    let n_dim = mu.len() as u32;
    MultivariateT { mu, psi, dof, n_dim }
  }

  fn gamma_div(&self, num: u32, div: u32) -> u32 {
    if num >= div {
      self.gamma_fn(num, 1, div)
    } else {
      self.gamma_fn(num, 1, 1) / self.gamma_fn(div, 1, 1)
    }
  }

  fn gamma(&self, x: u32) -> u32 {
    self.gamma_fn(x, 1, 1)
  }

  fn gamma_fn(&self, x: u32, agg: u32, stop_at: u32) -> u32 {
    match x {
      x if x == stop_at => agg,
      _ => self.gamma_fn(x - 1, agg * x, stop_at),
    }
  }

  pub(crate) fn posterior_predictive(&self, x: &Array1<f64>, sig: &Array2<f64>) -> f64 {
    // println!("X: {} | MU: {}", x, self.mu);
    // println!("SIG: {sig}");
    let deviation = x - &self.mu;
    // println!("DEVIATION: {deviation}");
    let deviation = deviation.into_owned();
    let proj = deviation
      .dot(&sig.solve(&deviation).unwrap());
      // .into_scalar();

    // println!("PROJ: {}", proj);

    let coeff = self.gamma_div((self.n_dim + self.dof) / 2, self.dof / 2) as f64
      * f64::from(self.dof).powf(-1.0 * (self.n_dim as f64 / 2.0 as f64))
      * (-std::f64::consts::LN_2 * self.n_dim as f64 / 2.0).exp()
      / f64::from(std::f64::consts::PI).powf(self.n_dim as f64 / 2.0);

    // println!("COEFF: {}", coeff);

    let sig_det = sig.det().unwrap();
    // println!("SIG DET: {sig_det}");
    let sig_det_inv_sqrt = sig_det.abs().sqrt().recip();
    // println!("SIG DET INV SQRT: {sig_det_inv_sqrt}");
    let the_rest = (1.0 + proj / f64::from(self.dof)).powf(-1.0 * (self.n_dim + self.dof) as f64 / 2.0);
    // println!("THE REST: {the_rest}");

    coeff * sig_det_inv_sqrt * the_rest
  }
}
