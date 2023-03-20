use std::collections::HashMap;
use ndarray::{Array1, Array2};
use crate::roque_stat::stream_crp::StreamCRP;
use crate::roque_stat::mvn::MVN;

pub struct Table {
  pub id: Vec<u8>,
  pub count: u16,
  pub component: MVN,
  pub alpha: f64,
  pub partition: Vec<Array1<f64>>,
}


impl Table {
  pub(crate) fn pp(&self, datum: &Array1<f64>) -> f64 {
    if self.count == 0 {
      self.alpha
    } else {
      self.component.pp(datum)
    }
  }

  pub(crate) fn draw(&self, n: usize) -> Array1<f64> {
    self.component.draw(n, false)
  }
}

pub(crate) trait BatchTable {
  fn seat(&mut self, datum: Array1<f64>);
}

impl BatchTable for Table {
  fn seat(&mut self, datum: Array1<f64>) {
    println!("Seating at {}", String::from_utf8(self.id.clone()).unwrap());
    self.count += 1;
    // let dat = datum.clone();
    self.partition.push(datum.clone());
    self.component.update(&datum.insert_axis(ndarray::Axis(0)));
  }
}
