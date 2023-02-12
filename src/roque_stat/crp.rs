use std::collections::HashMap;
use nalgebra::DVector;
use crate::roque_stat::stream_crp::StreamCRP;
use crate::roque_stat::table::Table;
use nalgebra::{DMatrix, RealField};
use uuid::Uuid;


pub(crate) trait CRP<T> {
  fn seat(&mut self, datum: DVector<f64>);
  fn reseat_all(&self, iterations: u64);
  fn with_tables(&self, new_tables: HashMap<Vec<u8>, T>) -> StreamCRP;

  fn dupe(&self, new_tables: HashMap<Vec<u8>, T>) -> StreamCRP;

  fn new_table_id(&self) -> Vec<u8> {
    Uuid::new_v4().to_string().into_bytes()
  }

  fn combine(&self, other: StreamCRP) -> StreamCRP;

  fn pp(&self, datum: DVector<f32>) -> DVector<f32>;

  fn draw(&self) -> Vec<DVector<f32>>;
}

