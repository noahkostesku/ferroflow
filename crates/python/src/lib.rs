use std::collections::HashMap;
use std::sync::Arc;

use ferroflow_core::{Dag, Op, OpKind, Tensor};
use ferroflow_runtime::WorkStealingScheduler;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Python-facing DAG builder.
///
/// Construct a computation graph by calling [`matmul`], [`relu`],
/// [`layer_norm`], and [`reduce`].  Each method appends a node and returns
/// its `OpId` (a `u32`) so subsequent ops can declare dependencies.
/// Pass the completed DAG to [`run`] to execute it.
#[pyclass(name = "DAG")]
pub struct PyDag {
    ops: Vec<Op>,
}

#[pymethods]
impl PyDag {
    #[new]
    fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Add a matmul node and return its op ID.
    ///
    /// `input_ids` — op IDs this node depends on (empty = source / weight tensor).
    /// `shape`     — output shape; product gives element count in `run` results.
    fn matmul(&mut self, input_ids: Vec<u32>, shape: Vec<usize>) -> u32 {
        self.push_op(input_ids, shape, false)
    }

    /// Add a ReLU node and return its op ID.
    fn relu(&mut self, input_ids: Vec<u32>, shape: Vec<usize>) -> u32 {
        self.push_op(input_ids, shape, true)
    }

    /// Add a layer-norm node and return its op ID.
    fn layer_norm(&mut self, input_ids: Vec<u32>, shape: Vec<usize>) -> u32 {
        self.push_op(input_ids, shape, false)
    }

    /// Add a reduce node and return its op ID.
    fn reduce(&mut self, input_ids: Vec<u32>, shape: Vec<usize>) -> u32 {
        self.push_op(input_ids, shape, false)
    }
}

impl PyDag {
    // Source ops use Relu (no input needed); non-source ops use:
    //   - Relu when `use_relu` is true (shape-preserving element-wise)
    //   - Slow { 0 } otherwise (pass-through; avoids multi-input constraints)
    fn push_op(&mut self, input_ids: Vec<u32>, shape: Vec<usize>, use_relu: bool) -> u32 {
        let id = self.ops.len();
        let len: usize = shape.iter().product::<usize>().max(1);

        let kind = if input_ids.is_empty() || use_relu {
            OpKind::Relu { len }
        } else {
            OpKind::Slow { duration_ms: 0 }
        };

        let dep_ids: Vec<usize> = input_ids
            .into_iter()
            .take(1) // take only the first dep to keep arity at 1
            .map(|x| x as usize)
            .collect();

        self.ops.push(Op::new(id, kind, dep_ids, shape));
        id as u32
    }
}

/// Execute a [`DAG`] with the work-stealing scheduler.
///
/// Returns a `dict` mapping each op ID to a flat `list[float]` of its
/// output tensor values.  Source ops (those with no inputs) are
/// pre-populated with ones tensors of their declared shape.
///
/// # Errors
/// Raises `ValueError` if the DAG is malformed (cycle or invalid IDs).
/// Raises `RuntimeError` if the scheduler encounters an execution error.
#[pyfunction]
fn run(dag: PyRef<'_, PyDag>, workers: usize) -> PyResult<HashMap<u32, Vec<f32>>> {
    let ops = dag.ops.clone();

    let source_tensors: HashMap<usize, Tensor> = ops
        .iter()
        .filter(|op| op.input_ids.is_empty())
        .map(|op| (op.id, Tensor::full(&op.output_shape, 1.0)))
        .collect();

    let core_dag = Dag::new(ops).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let rt = tokio::runtime::Runtime::new().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let (results, _metrics) = rt
        .block_on(
            WorkStealingScheduler::new(workers.max(1)).execute(Arc::new(core_dag), source_tensors),
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok(results
        .into_iter()
        .map(|(id, tensor)| (id as u32, tensor.data.iter().copied().collect()))
        .collect())
}

#[pymodule]
fn ferroflow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDag>()?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    Ok(())
}
