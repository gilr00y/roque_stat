mod roque_stat;

#[cfg(test)]
mod tests {
  use std::collections::HashMap;
  use crate::roque_stat::mvn::MVN;
  use crate::roque_stat::table::Table;
  use crate::roque_stat::batch_crp::BatchCRP;
  use crate::roque_stat::stream_crp::StreamCRP;


  #[test]
  fn adds_data_to_batch_crp() {
    let batch_crp = BatchCRP {
      data: vec![0.0, 1.1, 2.2, 3.3],
      alpha: 0.0,
      max_iterations: 0,
      tables: &mut HashMap::new()
    };

    assert_eq!(batch_crp.tables.len(), 1)
  }
}
