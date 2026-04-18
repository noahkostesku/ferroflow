use std::collections::{HashMap, HashSet, VecDeque};
use thiserror::Error;

use crate::op::{Op, OpId, OpKind};
use crate::tensor::Tensor;

/// Errors produced by [`Dag`] construction or traversal.
#[derive(Debug, Error)]
pub enum DagError {
    /// An `input_id` referenced an op that does not exist in the DAG.
    #[error("op {0} referenced as input but not found in DAG")]
    OpNotFound(OpId),

    /// The DAG contains a cycle and cannot be topologically sorted.
    #[error("cycle detected — DAG must be acyclic")]
    CycleDetected,
}

/// A directed acyclic graph of [`Op`] nodes.
///
/// Edges are encoded implicitly by each op's `input_ids`: an edge `A → B`
/// means B consumes the output tensor produced by A.
///
/// Op IDs must be 0..ops.len() and equal the op's index in the `ops` vec,
/// so any `OpId` can be used as a direct index.
pub struct Dag {
    /// All ops in the DAG, indexed by their `id`.
    pub ops: Vec<Op>,
    /// Precomputed: op_id → ids of ops that consume this op's output.
    successors: HashMap<OpId, Vec<OpId>>,
}

impl Dag {
    /// Constructs a `Dag` from a flat list of ops, validating that:
    /// - each `op.id == index` in the vec, and
    /// - all `input_ids` refer to ops that exist.
    ///
    /// # Errors
    /// Returns [`DagError::OpNotFound`] on invalid `input_ids`.
    pub fn new(ops: Vec<Op>) -> Result<Self, DagError> {
        // Validate: op.id must equal its index so we can use id as index.
        for (i, op) in ops.iter().enumerate() {
            if op.id != i {
                // Treat a mismatched id as if the expected-position id is missing.
                return Err(DagError::OpNotFound(op.id));
            }
        }

        let n = ops.len();

        // Validate input_ids and build successors.
        let mut successors: HashMap<OpId, Vec<OpId>> = (0..n).map(|i| (i, vec![])).collect();
        for op in &ops {
            for &dep_id in &op.input_ids {
                if dep_id >= n {
                    return Err(DagError::OpNotFound(dep_id));
                }
                successors.entry(dep_id).or_default().push(op.id);
            }
        }

        Ok(Self { ops, successors })
    }

    /// Returns a reference to op `id`, or `None` if out of range.
    pub fn get_op(&self, id: OpId) -> Option<&Op> {
        self.ops.get(id)
    }

    /// Returns ops not yet in `completed` whose every `input_id` is in `completed`.
    ///
    /// Source ops (empty `input_ids`) are always ready if not completed.
    pub fn ready_ops(&self, completed: &HashSet<OpId>) -> Vec<OpId> {
        self.ops
            .iter()
            .filter(|op| !completed.contains(&op.id))
            .filter(|op| op.input_ids.iter().all(|id| completed.contains(id)))
            .map(|op| op.id)
            .collect()
    }

    /// Returns a topological ordering of all op IDs (Kahn's algorithm).
    ///
    /// # Errors
    /// Returns [`DagError::CycleDetected`] if the graph contains a cycle.
    pub fn topological_sort(&self) -> Result<Vec<OpId>, DagError> {
        // in-degree = number of unprocessed dependencies
        let mut in_degree: Vec<usize> = self.ops.iter().map(|op| op.input_ids.len()).collect();

        let mut queue: VecDeque<OpId> = in_degree
            .iter()
            .enumerate()
            .filter(|(_, &d)| d == 0)
            .map(|(id, _)| id)
            .collect();

        let mut order = Vec::with_capacity(self.ops.len());

        while let Some(id) = queue.pop_front() {
            order.push(id);
            for &succ in self.successors.get(&id).map(Vec::as_slice).unwrap_or(&[]) {
                in_degree[succ] -= 1;
                if in_degree[succ] == 0 {
                    queue.push_back(succ);
                }
            }
        }

        if order.len() != self.ops.len() {
            return Err(DagError::CycleDetected);
        }
        Ok(order)
    }

    /// Number of ops in the DAG.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Returns `true` if the DAG contains no ops.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Builds a two-branch skewed DAG for benchmark skew-injection.
    ///
    /// Structure (total `2 + n_ops` ops):
    /// - Op 0: fast-branch source (no inputs, pre-populated as a source tensor).
    /// - Op 1: slow-branch source (no inputs, pre-populated as a source tensor).
    /// - Ops `2 .. 2 + half`: fast branch — each `Slow(1 ms)`, all depending only on op 0.
    /// - Ops `2 + half .. 2 + n_ops`: slow branch — each `Slow(slow_branch_factor ms)`,
    ///   all depending only on op 1.
    ///
    /// Returns `(dag, source_tensors)` where `source_tensors` contains the pre-built
    /// tensors for ops 0 and 1.
    ///
    /// # Errors
    /// Returns [`DagError`] if the DAG construction fails (should never occur for
    /// well-formed inputs).
    pub fn with_skew(
        n_ops: usize,
        slow_branch_factor: u64,
    ) -> Result<(Self, HashMap<OpId, Tensor>), DagError> {
        assert!(
            n_ops >= 2 && n_ops.is_multiple_of(2),
            "n_ops must be a positive even number"
        );
        let half = n_ops / 2;
        let mut ops: Vec<Op> = Vec::with_capacity(2 + n_ops);

        ops.push(Op::new(0, OpKind::Relu { len: 1 }, vec![], vec![1]));
        ops.push(Op::new(1, OpKind::Relu { len: 1 }, vec![], vec![1]));

        for i in 0..half {
            ops.push(Op::new(
                2 + i,
                OpKind::Slow { duration_ms: 1 },
                vec![0],
                vec![1],
            ));
        }
        for i in 0..half {
            ops.push(Op::new(
                2 + half + i,
                OpKind::Slow {
                    duration_ms: slow_branch_factor,
                },
                vec![1],
                vec![1],
            ));
        }

        let dag = Self::new(ops)?;
        let sources = HashMap::from([(0, Tensor::full(&[1], 1.0)), (1, Tensor::full(&[1], 1.0))]);
        Ok((dag, sources))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{Op, OpKind};

    fn matmul_op(id: OpId, input_ids: Vec<OpId>) -> Op {
        Op::new(
            id,
            OpKind::Matmul { m: 4, n: 4, k: 4 },
            input_ids,
            vec![4, 4],
        )
    }

    /// Build a 5-op DAG:
    ///
    ///   0 ──┐
    ///        ├─▶ 2 ──▶ 4
    ///   1 ──┘         ▲
    ///        3 ───────┘
    fn five_op_dag() -> Dag {
        Dag::new(vec![
            matmul_op(0, vec![]),
            matmul_op(1, vec![]),
            matmul_op(2, vec![0, 1]),
            matmul_op(3, vec![]),
            matmul_op(4, vec![2, 3]),
        ])
        .unwrap()
    }

    #[test]
    fn topo_sort_respects_dependencies() {
        let dag = five_op_dag();
        let order = dag.topological_sort().unwrap();

        // 0,1 must precede 2; 2 and 3 must precede 4.
        let pos: HashMap<OpId, usize> = order.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        assert!(pos[&0] < pos[&2]);
        assert!(pos[&1] < pos[&2]);
        assert!(pos[&2] < pos[&4]);
        assert!(pos[&3] < pos[&4]);
    }

    #[test]
    fn topo_sort_visits_all_ops() {
        let dag = five_op_dag();
        let mut order = dag.topological_sort().unwrap();
        order.sort_unstable();
        assert_eq!(order, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn ready_ops_empty_completed() {
        let dag = five_op_dag();
        let ready = dag.ready_ops(&HashSet::new());
        let mut ready_sorted = ready.clone();
        ready_sorted.sort_unstable();
        // Source ops: 0, 1, 3 have no inputs.
        assert_eq!(ready_sorted, vec![0, 1, 3]);
    }

    #[test]
    fn ready_ops_after_sources_complete() {
        let dag = five_op_dag();
        let completed: HashSet<OpId> = [0, 1, 3].into();
        let ready = dag.ready_ops(&completed);
        // 2 needs 0+1 (both done); 4 needs 2+3 (2 not done yet).
        assert_eq!(ready, vec![2]);
    }

    #[test]
    fn ready_ops_after_all_complete() {
        let dag = five_op_dag();
        let completed: HashSet<OpId> = (0..5).collect();
        assert!(dag.ready_ops(&completed).is_empty());
    }

    #[test]
    fn cycle_detected() {
        // 0 → 1 → 0
        let result = Dag::new(vec![matmul_op(0, vec![1]), matmul_op(1, vec![0])]);
        assert!(matches!(
            result.unwrap().topological_sort(),
            Err(DagError::CycleDetected)
        ));
    }

    #[test]
    fn invalid_input_id_rejected() {
        let result = Dag::new(vec![matmul_op(0, vec![99])]);
        assert!(matches!(result, Err(DagError::OpNotFound(99))));
    }
}
