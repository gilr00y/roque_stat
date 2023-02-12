use std::collections::HashMap;
use nalgebra::{DVector, dvector};
use crate::roque_stat::crp::CRP;
use crate::roque_stat::mvn::MVN;
use crate::roque_stat::table::{BatchTable, Table};


pub struct BatchCRP<'a> {
  pub data: Vec<f32>,
  pub alpha: f32,
  pub max_iterations: u32,
  pub tables: &'a mut HashMap<Vec<u8>, Box<BatchTable>>,
}

impl CRP<BatchTable> for BatchCRP<'_> {
  fn seat(&mut self, datum: DVector<f64>) {
    let new_table_id = self.new_table_id();
    let x = self.alpha;

    // Create a newBatchTable and temporarily add it to the
    let new_table = BatchTable {
      id: new_table_id.clone(),
      count: 0,
      component: MVN {
        mu: vec![0.0],
        psi: vec![vec![0.0]],
        k0: 0,
      },
      alpha: self.alpha,
      partition: vec![dvector![]],
    };

    self.tables.insert(new_table_id, Box::new(new_table));

    let table_pps: Vec<(&Vec<u8>, f64)> = self.tables.iter()
      .map(|(tbl_id, tbl)| {
        let pp = tbl.pp(&datum);
        let table_count = tbl.count;
        (tbl_id, pp)
      }).collect();




    // Select aBatchTable at which to seat the new record, then callBatchTable.seat
  }

  fn reseat_all(&self, iterations: u64) {
    todo!()
  }

  fn with_tables(&self, new_tables: HashMap<Vec<u8>,BatchTable>) -> crate::roque_stat::stream_crp::StreamCRP {
    todo!()
  }

  fn dupe(&self, new_tables: HashMap<Vec<u8>,BatchTable>) -> crate::roque_stat::stream_crp::StreamCRP {
    todo!()
  }

  fn combine(&self, other: crate::roque_stat::stream_crp::StreamCRP) -> crate::roque_stat::stream_crp::StreamCRP {
    todo!()
  }

  fn pp(&self, datum: DVector<f32>) -> DVector<f32> {
    todo!()
  }

  fn draw(&self) -> Vec<DVector<f32>> {
    todo!()
  }
}
