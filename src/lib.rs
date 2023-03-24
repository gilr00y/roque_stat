pub mod roque_stat;

#[cfg(test)]
mod tests {
  use std::collections::HashMap;
  use ndarray::{Array1, Array2};
  // use crate::roque_stat::mvn::MVN;
  use crate::roque_stat::table::Table;
  use crate::roque_stat::batch_crp::BatchCRP;
  use crate::roque_stat::crp::CRP;
  // use crate::roque_stat::stream_crp::StreamCRP;


  #[test]
  fn adds_data_to_batch_crp() {
    let mut batch_crp = BatchCRP {
      alpha: 0.1,
      max_iterations: 0,
      tables: Box::new(HashMap::new()),
      psi_scale: Array1::from(vec![1.0, 1.0, 1.0])
    };
    let datum = Array1::from(vec![0.1, 0.2, 0.3]);
    batch_crp.seat(datum);

    // Check that a table was created.
    assert_eq!(batch_crp.tables.len(), 1);

    // Check that our table actually has data.
    let tables: Vec<&Box<Table>> = batch_crp.tables.values().collect();
    assert_eq!(tables.first().unwrap().count, 1);
  }

  #[test]
  fn adds_similar_data_to_existing_table() {
    let mut batch_crp = BatchCRP {
      alpha: 0.01,
      max_iterations: 0,
      tables: Box::new(HashMap::new()),
      psi_scale: Array1::from(vec![1.0, 1.0, 1.0])
    };
    let datum = Array1::from(vec![0.1, 0.2, 0.3]);
    batch_crp.seat(datum.clone());
    batch_crp.seat(datum);
    assert_eq!(batch_crp.tables.len(), 1);

    // Check that our table actually has data.
    let tables: Vec<&Box<Table>> = batch_crp.tables.values().collect();
    for table in tables {
      assert_eq!(table.count, 2);
    }
  }

  #[test]
  fn adds_disparate_data_to_new_table() {
    let mut batch_crp = BatchCRP {
      alpha: 0.1,
      max_iterations: 0,
      tables: Box::new(HashMap::new()),
      psi_scale: Array1::from(vec![1.0, 1.0, 1.0])
    };
    let datum = Array1::from(vec![0.1, 0.2, 0.3]);
    let new_datum = Array1::from(vec![0.1, 0.2, 0.3]);//Array1::from(vec![15.0, 14.1, 11.9]);
    batch_crp.seat(datum);
    batch_crp.seat(new_datum);

    assert_eq!(batch_crp.tables.len(), 2);

    // Check that our table actually has data.
    let tables: Vec<&Box<Table>> = batch_crp.tables.values().collect();
    for table in tables {
      assert_eq!(table.count, 1);
    }
  }

  #[test]
  fn draws_sample_from_crp() {
    let mut batch_crp = BatchCRP {
      alpha: 0.001,
      max_iterations: 0,
      tables: Box::new(HashMap::new()),
      psi_scale: Array1::from(vec![1.0, 1.0, 1.0])
    };
    let datum = Array1::from(vec![0.1, 0.2, 0.3]);
    batch_crp.seat(datum.clone());
    batch_crp.seat(datum.clone());
    assert_eq!(batch_crp.draw(1).first().unwrap().len(), datum.len());
    assert_eq!(batch_crp.draw(1).first().unwrap().len(), 3);
    // assert_eq!(batch_crp.draw(1).first().unwrap().nrows(), datum.nrows());
  }

  fn returns_high_posterior_probability_when_near_table_mean() {
    todo!()
  }
}
