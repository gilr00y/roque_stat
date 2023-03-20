use std::collections::HashMap;
use ndarray::Array1;
use crate::roque_stat::stream_crp::StreamCRP;
use crate::roque_stat::table::Table;
use uuid::Uuid;


pub trait CRP<T> {
  fn seat(&mut self, datum: Array1<f64>);
  fn reseat_all(&self, iterations: u64);
  // fn with_tables(&self, new_tables: HashMap<Vec<u8>, T>) -> StreamCRP;

  // fn dupe(&self, new_tables: HashMap<Vec<u8>, T>) -> StreamCRP;

  fn new_table_id(&self) -> Vec<u8> {
    Uuid::new_v4().to_string().into_bytes()
  }

  fn combine(&self, other: Self) -> Self;

  fn pp(&self, datum: Array1<f64>) -> f64;

  fn draw(&self, n: usize) -> Vec<Array1<f64>>;
}

