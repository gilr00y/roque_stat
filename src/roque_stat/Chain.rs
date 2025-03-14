use std::collections::HashMap;
use ndarray::{s, Array1};
use crate::roque_stat::batch_crp::BatchCRP;
use crate::roque_stat::crp::CRP;

pub struct Chain {
    pub permutations: HashMap<(usize, usize, usize), BatchCRP>,
}

impl Chain {
    pub fn new(data: Vec<Array1<f64>>) -> Self {
        let n_dims = data.first().unwrap().shape()[0];
        let permutations = Self::get_permutations(n_dims);

        let mut component_crps = HashMap::new();
        for p in permutations {
            let crp = BatchCRP {
                alpha: 0.01,
                max_iterations: 10000,
                tables: Box::new(HashMap::new()),
                psi_scale: Array1::from(vec![1.0, 1.0, 1.0]),
            };
            component_crps.insert(p, crp);
        }

        Chain {
            permutations: component_crps
        }
    }

    pub fn seat(&mut self, datum: &Array1<f64>) {
        let keys = self.permutations.keys().cloned().collect::<Vec<_>>();
        for dims in keys {
            if let Some(crp) = self.permutations.get_mut(&dims) {
                let mut partial = Array1::zeros(3);
                partial[0] = datum[dims.0];
                partial[1] = datum[dims.1];
                partial[2] = datum[dims.2];

                crp.seat(partial);
            }
        }
    }

    fn get_permutations(dims: usize) -> Vec<(usize, usize, usize)> {
        let mut result = Vec::new();

        // We need n ≥ 3 to form any 3-tuples
        if dims < 3 {
            return result;
        }

        // Generate all combinations where i < j < k to ensure uniqueness
        // This ensures we get each combination exactly once
        for i in 0..dims {
            for j in i + 1..dims {
                for k in j + 1..dims {
                    result.push((i, j, k));
                }
            }
        }

        result
    }
}