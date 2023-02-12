use std::collections::HashMap;
use nalgebra::DVector;
use crate::roque_stat::crp::CRP;
use crate::roque_stat::table::StreamTable;

pub struct StreamCRP {
  pub alpha: f32,
  pub max_iterations: u32,
  pub tables: HashMap<u16, String>,
}

impl CRP<StreamTable> for StreamCRP {
  fn seat(&mut self, datum: DVector<f64>) {
    todo!()
  }

  fn reseat_all(&self, iterations: u64) {
    todo!()
  }

  fn with_tables(&self, new_tables: HashMap<Vec<u8>, StreamTable>) -> StreamCRP {
    todo!()
  }

  fn dupe(&self, new_tables: HashMap<Vec<u8>, StreamTable>) -> StreamCRP {
    todo!()
  }

  fn combine(&self, other: StreamCRP) -> StreamCRP {
    todo!()
  }

  fn pp(&self, datum: DVector<f32>) -> DVector<f32> {
    todo!()
  }

  fn draw(&self) -> Vec<DVector<f32>> {
    todo!()
  }
}
