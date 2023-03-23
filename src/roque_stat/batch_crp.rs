
use std::collections::HashMap;
use ndarray::{Array1, Array2};
use rand;
use rand::Rng;
use crate::roque_stat::crp::CRP;
use crate::roque_stat::mvn::MVN;
use crate::roque_stat::stream_crp::StreamCRP;
use crate::roque_stat::table::{BatchTable, Table};


pub struct BatchCRP {
  pub alpha: f64,
  pub max_iterations: u32,
  pub tables: Box<HashMap<Vec<u8>, Box<Table>>>,
  pub psi_scale: Array1<f64>,
}

impl BatchCRP {
  fn get_table_pps(&self, datum: &Array1<f64>) -> Vec<(&Vec<u8>, f64, u16)> {
    self.tables.iter()
      .map(|(tbl_id, tbl)| {
        let weighted_pp = tbl.pp(datum);
        println!("TBLE_PP: {}", weighted_pp);
        let table_count = tbl.count;
        println!("TBL_COUNT: {}", tbl.count);
        // New table has table_count = 0, so the pp is also 0.
        (tbl_id, weighted_pp * table_count as f64, tbl.count)
      }).collect()
  }

  fn clean_up(&mut self, seated_tbl_id: Vec<u8>, new_tbl_id: Vec<u8>) {
    if seated_tbl_id != new_tbl_id {
      self.tables.remove(&new_tbl_id);
    }
  }

  fn seat_datum(&mut self, datum: Array1<f64>) -> (Vec<u8>, Vec<u8>) {
    let new_table_id = self.new_table_id();
    let x = self.alpha;

    // Create a new table and temporarily add it.
    let new_table = Table {
      id: new_table_id.clone(),
      count: 0,
      component: MVN {
        n_dim: datum.len(),
        mu: datum.clone(),
        psi: Array2::from_diag(&self.psi_scale),
        k0: 0,
      },
      alpha: self.alpha,
      partition: vec![],
    };

    self.tables.insert(new_table_id.clone(), Box::new(new_table));

    let table_pps: Vec<(&Vec<u8>, f64, u16)> = self.get_table_pps(&datum);

    let sum_pp = table_pps.iter().fold(0.0, |acc, el| {
      let pp = el.1;
      acc + pp
    }) + self.alpha;


    let mut positional_collector = 0.0;
    // TODO: Cache these calculations
    let mut normalized_table_pps: Vec<(&Vec<u8>, f64)> = table_pps.iter()
      .map(|el| {
        match el {
          (tbl_id, pp, tbl_count) => {
            let normalized_pp = if *tbl_count == 0 {
              // println!("TBL COUNT IS 0, so PP is being set to {}", self.alpha / sum_pp);
              // println!("COunt = 0, ALpha: {}, Sum_pp: {}", self.alpha, sum_pp);

              self.alpha / sum_pp
            } else {
              // println!("TBL COUNT NOT 0, SO PP CALCULATED as {}", *pp / sum_pp);
              // println!("COunt = 0, ALpha: {}, Sum_pp: {}", self.alpha, sum_pp);
              *pp / sum_pp
            };
            // println!("NORMALIZED_PP: {}", normalized_pp);
            positional_collector += normalized_pp;
            (*tbl_id, positional_collector)
          }
        }
      })
      .collect();

    // Select random point in range (0, 1).
    let select: f64 = rand::thread_rng().gen_range(0.0..1.0);

    // Create thresholds based on each normalized_pp.
    let select_tbl_id = {
      let mut _tmp_id = vec![] as Vec<u8>;

      for (tbl_id, max_thresh) in normalized_table_pps {
        // println!("TABLE ID: {}", String::from_utf8_lossy(tbl_id));
        // println!("SELECT: {}", select);
        // println!("max thresh: {}", max_thresh);
        if select < max_thresh {
          _tmp_id = tbl_id.clone();
          break;
        }
      }
      _tmp_id
    };

    // Assign datum to that table.
    let mut table = self.tables.get_mut(select_tbl_id.clone().as_slice()).unwrap();
    table.seat(datum.clone());
    (select_tbl_id, new_table_id)
  }
}

impl CRP<Table> for BatchCRP {
  fn seat(&mut self, datum: Array1<f64>) {
    let (seated_tbl_id, new_tbl_id) = self.seat_datum(datum);
    self.clean_up(seated_tbl_id, new_tbl_id);
  }

  fn reseat_all(&self, iterations: u64) {
    todo!()
  }

  // fn dupe(&self, new_tables: HashMap<Vec<u8>, dyn BatchTable>) -> StreamCRP {
  //   todo!()
  // }

  // fn with_tables(&self, new_tables: HashMap<Vec<u8>, dyn BatchTable>) -> Self {
  //   todo!()
  // }

  // fn dupe(&self, new_tables: HashMap<Vec<u8>, dyn BatchTable>) -> crate::roque_stat::stream_crp::StreamCRP {
  //   todo!()
  // }

  fn combine(&self, other: Self) -> Self {
    todo!()
  }

  fn pp(&self, datum: Array1<f64>) -> f64 {
    // TODO: Rethink this...
    self.tables.values().into_iter().map(|tbl| tbl.pp(&datum)).sum()
  }

  fn draw(&self, n: usize) -> Vec<Array1<f64>> {
    // Step 1: Get full count.
    let mut samples: Vec<Array1<f64>> = vec![];
    for _ in 0..n {
      let total_count: u16 = self.tables.values().into_iter().map(|tbl| tbl.count).sum();
      // Step 2: Probabilistically select the table based on random/count
      let rand_select: f64 = rand::thread_rng().gen_range(0.0..1.0);
      let tbl_id = {
        let mut _tmp_id = vec![];
        let mut min_range = 0.0;
        for table in self.tables.values() {
          let max_range = min_range + table.count as f64 / total_count as f64;
          if rand_select < max_range {
            _tmp_id = table.id.clone();
            break;
          }
          min_range = max_range;
        }
        _tmp_id
      };
      let selected_table = self.tables.get(&tbl_id).unwrap();

      // Step 3: Draw from table and add to samples.
      samples.push(selected_table.draw(1))
    }
    samples
  }
}
