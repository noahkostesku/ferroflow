use std::collections::HashMap;

use ferroflow_core::{ops::execute_op, Dag, DagError, OpError, OpId, Tensor};
use thiserror::Error;

/// Errors returned by [`SequentialExecutor::execute`].
#[derive(Debug, Error)]
pub enum ExecutorError {
    /// Topological sort failed (cycle in the DAG).
    #[error("DAG error: {0}")]
    Dag(#[from] DagError),

    /// An input tensor required by an op was not found in the tensor store.
    ///
    /// This means the caller did not provide a source tensor, or a bug in the
    /// executor left an op's output unrecorded.
    #[error("op {0}: missing input tensor for op {1}")]
    MissingInput(OpId, OpId),

    /// Execution of a specific op failed.
    #[error("op {op_id} ({op_name}) failed: {source}")]
    OpFailed {
        op_id: OpId,
        op_name: &'static str,
        #[source]
        source: OpError,
    },
}

/// Single-threaded baseline executor.
///
/// Walks the DAG in topological order and executes each op sequentially,
/// accumulating output tensors in a `HashMap<OpId, Tensor>`.
///
/// This is the reference implementation that all distributed schedulers are
/// benchmarked against.
pub struct SequentialExecutor;

impl SequentialExecutor {
    /// Executes `dag` starting from the provided `source_tensors`.
    ///
    /// `source_tensors` must contain a [`Tensor`] for every source op (an op
    /// with no `input_ids`). All other tensors are computed on demand.
    ///
    /// Returns the complete `HashMap<OpId, Tensor>` after the full DAG has run.
    ///
    /// # Errors
    /// Returns [`ExecutorError`] on DAG cycles, missing source tensors, or op
    /// execution failures.
    pub fn execute(
        dag: &Dag,
        source_tensors: HashMap<OpId, Tensor>,
    ) -> Result<HashMap<OpId, Tensor>, ExecutorError> {
        let order = dag.topological_sort()?;
        let mut store: HashMap<OpId, Tensor> = source_tensors;

        for op_id in order {
            // Source ops are already in the store — skip execution.
            if store.contains_key(&op_id) {
                continue;
            }

            let op = dag.get_op(op_id).expect("topological_sort only yields valid ids");

            let inputs: Vec<&Tensor> = op
                .input_ids
                .iter()
                .map(|&dep_id| {
                    store
                        .get(&dep_id)
                        .ok_or(ExecutorError::MissingInput(op_id, dep_id))
                })
                .collect::<Result<_, _>>()?;

            let op_name = op_kind_name(&op.kind);
            let output = execute_op(op, &inputs)
                .map_err(|source| ExecutorError::OpFailed { op_id, op_name, source })?;

            store.insert(op_id, output);
        }

        Ok(store)
    }
}

fn op_kind_name(kind: &ferroflow_core::OpKind) -> &'static str {
    match kind {
        ferroflow_core::OpKind::Matmul { .. } => "matmul",
        ferroflow_core::OpKind::Relu { .. } => "relu",
        ferroflow_core::OpKind::LayerNorm { .. } => "layer_norm",
        ferroflow_core::OpKind::Reduce { .. } => "reduce",
    }
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ferroflow_core::{Op, OpKind};

    /// Builds the linear chain  src → relu → relu → relu  and verifies the
    /// final tensor still has all non-negative values.
    #[test]
    fn linear_relu_chain() {
        let ops = vec![
            Op::new(0, OpKind::Relu { len: 4 }, vec![], vec![4]),   // source
            Op::new(1, OpKind::Relu { len: 4 }, vec![0], vec![4]),
            Op::new(2, OpKind::Relu { len: 4 }, vec![1], vec![4]),
            Op::new(3, OpKind::Relu { len: 4 }, vec![2], vec![4]),
        ];
        let dag = Dag::new(ops).unwrap();

        let source = Tensor::from_shape_vec(&[4], vec![-3., -1., 0., 2.]).unwrap();
        let mut sources = HashMap::new();
        sources.insert(0usize, source);

        let results = SequentialExecutor::execute(&dag, sources).unwrap();

        let out: Vec<f32> = results[&3].data.iter().copied().collect();
        assert!(out.iter().all(|&v| v >= 0.0), "relu output must be non-negative: {out:?}");
        assert_eq!(out[3], 2.0);
    }

    /// Exercises a diamond dependency:  A ──┐
    ///                                       ├─▶ matmul
    ///                                  B ──┘
    #[test]
    fn diamond_matmul() {
        let ops = vec![
            Op::new(0, OpKind::Matmul { m: 2, n: 2, k: 2 }, vec![], vec![2, 2]),  // A source
            Op::new(1, OpKind::Matmul { m: 2, n: 2, k: 2 }, vec![], vec![2, 2]),  // B source
            Op::new(2, OpKind::Matmul { m: 2, n: 2, k: 2 }, vec![0, 1], vec![2, 2]),
        ];
        let dag = Dag::new(ops).unwrap();

        let mut sources = HashMap::new();
        // A = I, B = [[1,2],[3,4]] → A·B = [[1,2],[3,4]]
        sources.insert(0usize, Tensor::from_shape_vec(&[2, 2], vec![1., 0., 0., 1.]).unwrap());
        sources.insert(1usize, Tensor::from_shape_vec(&[2, 2], vec![1., 2., 3., 4.]).unwrap());

        let results = SequentialExecutor::execute(&dag, sources).unwrap();
        let flat: Vec<f32> = results[&2].data.iter().copied().collect();
        assert_eq!(flat, vec![1., 2., 3., 4.]);
    }

    /// Every op in the result map should be present.
    #[test]
    fn all_ops_produce_outputs() {
        let n = 5usize;
        let ops: Vec<Op> = (0..n)
            .map(|i| {
                let inputs = if i == 0 { vec![] } else { vec![i - 1] };
                Op::new(i, OpKind::Relu { len: 8 }, inputs, vec![8])
            })
            .collect();
        let dag = Dag::new(ops).unwrap();

        let mut sources = HashMap::new();
        sources.insert(0usize, Tensor::full(&[8], 1.0));

        let results = SequentialExecutor::execute(&dag, sources).unwrap();
        assert_eq!(results.len(), n);
    }
}
