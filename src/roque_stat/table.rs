use std::collections::HashMap;
use nalgebra::DVector;
use crate::roque_stat::stream_crp::StreamCRP;
use crate::roque_stat::mvn::MVN;

// pub struct Table {
//   pub id: Vec<u8>,
//   pub count: u16,
//   pub component: MVN,
//   pub alpha: f32,
// }

pub struct BatchTable {
  pub id: Vec<u8>,
  pub count: u16,
  pub component: MVN,
  pub alpha: f32,
  pub partition: Vec<DVector<f64>>
}

pub struct StreamTable {
  pub id: Vec<u8>,
  pub count: u16,
  pub component: MVN,
  pub alpha: f32,
}

pub(crate) trait Table {
  fn pp(&self, datum: &DVector<f64>) -> f64;

}

impl Table for BatchTable {
  fn pp(&self, datum: &DVector<f64>) -> f64 {
    1.23
  }
}
