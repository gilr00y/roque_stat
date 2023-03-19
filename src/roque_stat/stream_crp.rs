use std::collections::HashMap;
use ndarray::Array1;
use crate::roque_stat::crp::CRP;
use crate::roque_stat::table::Table;
// use crate::roque_stat::table::StreamTable;

pub struct StreamCRP {
  pub alpha: f64,
  pub max_iterations: u32,
  pub tables: HashMap<u16, String>,
}

impl CRP<Table> for StreamCRP {
  fn seat(&mut self, datum: Array1<f64>) {
    todo!()
  }

  fn reseat_all(&self, iterations: u64) {
    todo!()
  }

  // fn with_tables(&self, new_tables: HashMap<Vec<u8>, Table>) -> StreamCRP {
  //   todo!()
  // }
  //
  // fn dupe(&self, new_tables: HashMap<Vec<u8>, Table>) -> StreamCRP {
  //   todo!()
  // }

  fn combine(&self, other: StreamCRP) -> StreamCRP {
    todo!()
  }

  fn pp(&self, datum: Array1<f64>) -> f64 {
    todo!()
  }

  fn draw(&self) -> Vec<Array1<f64>> {
    todo!()
  }
}
