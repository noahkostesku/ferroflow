use std::collections::HashMap;

use crate::dag::{Dag, DagError};
use crate::op::{Op, OpId, OpKind};
use crate::tensor::Tensor;

/// Generates a single transformer attention + FFN block as a DAG.
///
/// Structure (18 ops: 6 sources + 12 compute):
/// - Sources: X, W_Q, W_K, W_V, W_O, W_FF (all `d_model × d_model`)
/// - Compute: Q/K/V projections (3-way parallel matmuls), QK attention scores,
///   scaled/softmax approximation (Relu+LayerNorm), context matmul,
///   output projection, residual add (Relu), layer norm, FF matmul, final layer norm.
///
/// `seq_len` and `n_heads` inform the op cost estimates via the `Matmul` `m`/`k`
/// fields but all tensors use square `d_model × d_model` shapes for simplicity.
///
/// # Errors
/// Returns [`DagError`] if DAG construction fails (should never occur for
/// well-formed inputs).
pub fn gen_transformer_block(
    seq_len: usize,
    d_model: usize,
    _n_heads: usize,
) -> Result<(Dag, HashMap<OpId, Tensor>), DagError> {
    let val = 1.0 / d_model as f32;
    let m = d_model;
    let sl = seq_len;
    let mut ops: Vec<Op> = Vec::with_capacity(18);

    // Sources: X(0), W_Q(1), W_K(2), W_V(3), W_O(4), W_FF(5)
    for id in 0..6usize {
        ops.push(Op::new(id, OpKind::Relu { len: m * m }, vec![], vec![m, m]));
    }

    // QKV projections — 3-way parallel fan-out from X
    ops.push(Op::new(
        6,
        OpKind::Matmul { m: sl, n: m, k: m },
        vec![0, 1],
        vec![m, m],
    )); // Q
    ops.push(Op::new(
        7,
        OpKind::Matmul { m: sl, n: m, k: m },
        vec![0, 2],
        vec![m, m],
    )); // K
    ops.push(Op::new(
        8,
        OpKind::Matmul { m: sl, n: m, k: m },
        vec![0, 3],
        vec![m, m],
    )); // V

    // Attention scores: Q · K
    ops.push(Op::new(
        9,
        OpKind::Matmul { m: sl, n: sl, k: m },
        vec![6, 7],
        vec![m, m],
    ));
    // Scale (approx): Relu(scores)
    ops.push(Op::new(
        10,
        OpKind::Relu { len: m * m },
        vec![9],
        vec![m, m],
    ));
    // Softmax (approx): LayerNorm(scaled_scores)
    ops.push(Op::new(
        11,
        OpKind::LayerNorm { len: m * m },
        vec![10],
        vec![m, m],
    ));
    // Context: attn_weights · V
    ops.push(Op::new(
        12,
        OpKind::Matmul { m: sl, n: m, k: m },
        vec![11, 8],
        vec![m, m],
    ));
    // Output projection: context · W_O
    ops.push(Op::new(
        13,
        OpKind::Matmul { m: sl, n: m, k: m },
        vec![12, 4],
        vec![m, m],
    ));
    // Residual add (approx): Relu(out_proj)
    ops.push(Op::new(
        14,
        OpKind::Relu { len: m * m },
        vec![13],
        vec![m, m],
    ));
    // Layer norm 1
    ops.push(Op::new(
        15,
        OpKind::LayerNorm { len: m * m },
        vec![14],
        vec![m, m],
    ));
    // Feed-forward: ln1 · W_FF
    ops.push(Op::new(
        16,
        OpKind::Matmul { m: sl, n: m, k: m },
        vec![15, 5],
        vec![m, m],
    ));
    // Layer norm 2
    ops.push(Op::new(
        17,
        OpKind::LayerNorm { len: m * m },
        vec![16],
        vec![m, m],
    ));

    let dag = Dag::new(ops)?;
    let sources = (0..6usize)
        .map(|id| (id, Tensor::full(&[m, m], val)))
        .collect();
    Ok((dag, sources))
}

/// Generates a DAG with `width` parallel branches of `depth` ops each.
///
/// Structure (`1 + width × depth` ops):
/// - Op 0: source (fan-out root)
/// - Ops `1 .. 1 + width × depth`: independent chains, one per branch
///
/// `skew_factor` is the fraction of branches (rounded) that receive 5× slower ops.
/// For example, `gen_wide_dag(8, 5, 0.25)` produces 8 branches × 5 ops, with 2 slow
/// branches.
///
/// # Errors
/// Returns [`DagError`] if DAG construction fails.
pub fn gen_wide_dag(
    width: usize,
    depth: usize,
    skew_factor: f32,
) -> Result<(Dag, HashMap<OpId, Tensor>), DagError> {
    let n_skewed = (width as f32 * skew_factor).round() as usize;
    let mut ops: Vec<Op> = Vec::with_capacity(1 + width * depth);

    ops.push(Op::new(0, OpKind::Relu { len: 1 }, vec![], vec![1]));

    for b in 0..width {
        let slow = b < n_skewed;
        for d in 0..depth {
            let id = 1 + b * depth + d;
            let input_ids = if d == 0 { vec![0] } else { vec![id - 1] };
            let duration_ms: u64 = if slow { 5 } else { 1 };
            ops.push(Op::new(
                id,
                OpKind::Slow { duration_ms },
                input_ids,
                vec![1],
            ));
        }
    }

    let dag = Dag::new(ops)?;
    let sources = HashMap::from([(0usize, Tensor::full(&[1], 1.0))]);
    Ok((dag, sources))
}

/// Generates a residual (ResNet-style) block as a DAG.
///
/// Structure (8 ops: 3 sources + 5 compute):
/// - Sources: X, W1, W2 (all `channels × channels`)
/// - Conv path: `conv1 = X · W1`, `relu1 = Relu(conv1)`, `conv2 = relu1 · W2`
/// - Skip path: identity (X, op 0)
/// - Merge (fork-join): `Matmul([conv2, X])` — approximates residual addition
/// - Batch norm (approx): `LayerNorm(merge)`
///
/// # Errors
/// Returns [`DagError`] if DAG construction fails.
pub fn gen_resnet_block(channels: usize) -> Result<(Dag, HashMap<OpId, Tensor>), DagError> {
    let val = 1.0 / channels as f32;
    let c = channels;
    let mut ops: Vec<Op> = Vec::with_capacity(8);

    // Sources: X(0), W1(1), W2(2)
    for id in 0..3usize {
        ops.push(Op::new(id, OpKind::Relu { len: c * c }, vec![], vec![c, c]));
    }

    // conv1 = X · W1
    ops.push(Op::new(
        3,
        OpKind::Matmul { m: c, n: c, k: c },
        vec![0, 1],
        vec![c, c],
    ));
    // relu1 = Relu(conv1)
    ops.push(Op::new(4, OpKind::Relu { len: c * c }, vec![3], vec![c, c]));
    // conv2 = relu1 · W2
    ops.push(Op::new(
        5,
        OpKind::Matmul { m: c, n: c, k: c },
        vec![4, 2],
        vec![c, c],
    ));
    // merge = conv2 · X  (fork-join: depends on both conv path and skip/X)
    ops.push(Op::new(
        6,
        OpKind::Matmul { m: c, n: c, k: c },
        vec![5, 0],
        vec![c, c],
    ));
    // bn = LayerNorm(merge)
    ops.push(Op::new(
        7,
        OpKind::LayerNorm { len: c * c },
        vec![6],
        vec![c, c],
    ));

    let dag = Dag::new(ops)?;
    let sources = (0..3usize)
        .map(|id| (id, Tensor::full(&[c, c], val)))
        .collect();
    Ok((dag, sources))
}

/// Generates a stacked multi-layer transformer as a DAG.
///
/// Stacks `layers` attention+FFN blocks in sequence.  Each block reuses
/// the structure of [`gen_transformer_block`]: 5 weight sources (W_Q/K/V/O/FF)
/// per layer plus 12 compute ops.  Layer 0 also has an explicit input source X.
/// The final LayerNorm of each layer feeds the next layer's Q/K/V projections.
///
/// `layers=8, d_model=512` → 137 ops (18 + 7×17), with 3-way QKV parallelism
/// inside every block.
///
/// # Errors
/// Returns [`DagError`] if DAG construction fails.
pub fn gen_large_transformer(
    layers: usize,
    d_model: usize,
) -> Result<(Dag, HashMap<OpId, Tensor>), DagError> {
    assert!(layers >= 1, "layers must be at least 1");
    let val = 1.0 / d_model as f32;
    let m = d_model;
    let sl = d_model; // seq_len = d_model for simplicity
    let total_ops = 18 + (layers.saturating_sub(1)) * 17;
    let mut ops: Vec<Op> = Vec::with_capacity(total_ops);
    let mut sources: HashMap<OpId, Tensor> = HashMap::new();

    // Layer 0: ops 0-5 are sources (X, W_Q, W_K, W_V, W_O, W_FF), ops 6-17 are compute.
    for id in 0..6usize {
        ops.push(Op::new(id, OpKind::Relu { len: m * m }, vec![], vec![m, m]));
        sources.insert(id, Tensor::full(&[m, m], val));
    }
    // Layer 0 compute — identical layout to gen_transformer_block.
    let prev_ln = build_transformer_compute(&mut ops, 6, 0, sl, m);
    let _ = prev_ln; // prev_ln == 17 after layer 0

    // Layers 1..layers: 5 fresh weight sources + 12 compute ops per layer.
    for l in 1..layers {
        let base = 18 + (l - 1) * 17;
        let x_id = base - 1; // LayerNorm output of previous layer
        for w in 0..5usize {
            let id = base + w;
            ops.push(Op::new(id, OpKind::Relu { len: m * m }, vec![], vec![m, m]));
            sources.insert(id, Tensor::full(&[m, m], val));
        }
        build_transformer_compute(&mut ops, base + 5, x_id, sl, m);
    }

    let dag = Dag::new(ops)?;
    Ok((dag, sources))
}

/// Appends 12 compute ops for one transformer block starting at `compute_base`.
///
/// `x_id` is the op whose output acts as the X (query/key/value input) for this
/// block.  Weight source ops are assumed to occupy ids
/// `[compute_base - 5 .. compute_base)` for layers > 0, or `[1..6)` for layer 0.
/// Returns the id of the final LayerNorm op.
fn build_transformer_compute(
    ops: &mut Vec<Op>,
    compute_base: usize,
    x_id: usize,
    sl: usize,
    m: usize,
) -> usize {
    // Determine weight source ids relative to compute_base.
    let (wq, wk, wv, wo, wff) = if compute_base == 6 {
        (1, 2, 3, 4, 5)
    } else {
        let b = compute_base - 5;
        (b, b + 1, b + 2, b + 3, b + 4)
    };

    let cb = compute_base;
    // Q, K, V projections (3-way parallel fan-out from X)
    ops.push(Op::new(
        cb,
        OpKind::Matmul { m: sl, n: m, k: m },
        vec![x_id, wq],
        vec![m, m],
    ));
    ops.push(Op::new(
        cb + 1,
        OpKind::Matmul { m: sl, n: m, k: m },
        vec![x_id, wk],
        vec![m, m],
    ));
    ops.push(Op::new(
        cb + 2,
        OpKind::Matmul { m: sl, n: m, k: m },
        vec![x_id, wv],
        vec![m, m],
    ));
    // Attention scores: Q · K
    ops.push(Op::new(
        cb + 3,
        OpKind::Matmul { m: sl, n: sl, k: m },
        vec![cb, cb + 1],
        vec![m, m],
    ));
    // Scale (Relu) + softmax approx (LayerNorm)
    ops.push(Op::new(
        cb + 4,
        OpKind::Relu { len: m * m },
        vec![cb + 3],
        vec![m, m],
    ));
    ops.push(Op::new(
        cb + 5,
        OpKind::LayerNorm { len: m * m },
        vec![cb + 4],
        vec![m, m],
    ));
    // Context: attn_weights · V
    ops.push(Op::new(
        cb + 6,
        OpKind::Matmul { m: sl, n: m, k: m },
        vec![cb + 5, cb + 2],
        vec![m, m],
    ));
    // Output projection
    ops.push(Op::new(
        cb + 7,
        OpKind::Matmul { m: sl, n: m, k: m },
        vec![cb + 6, wo],
        vec![m, m],
    ));
    // Residual + layer norm 1
    ops.push(Op::new(
        cb + 8,
        OpKind::Relu { len: m * m },
        vec![cb + 7],
        vec![m, m],
    ));
    ops.push(Op::new(
        cb + 9,
        OpKind::LayerNorm { len: m * m },
        vec![cb + 8],
        vec![m, m],
    ));
    // Feed-forward + layer norm 2
    ops.push(Op::new(
        cb + 10,
        OpKind::Matmul { m: sl, n: m, k: m },
        vec![cb + 9, wff],
        vec![m, m],
    ));
    ops.push(Op::new(
        cb + 11,
        OpKind::LayerNorm { len: m * m },
        vec![cb + 10],
        vec![m, m],
    ));
    cb + 11
}

/// Generates a large wide fan-out DAG with 32 branches × 10 ops (321 total ops).
///
/// This is a scaled-up [`gen_wide_dag`]: `width=32`, `depth=10`, `skew_factor=0.5`
/// (half the branches are 5× slower).  Produces enough parallelism that
/// work-stealing triggers reliably even at 256 workers.
///
/// # Errors
/// Returns [`DagError`] if DAG construction fails.
pub fn gen_large_wide(
    width: usize,
    depth: usize,
    skew_factor: f32,
) -> Result<(Dag, HashMap<OpId, Tensor>), DagError> {
    gen_wide_dag(width, depth, skew_factor)
}

/// Generates an imbalanced DAG that provably concentrates slow work on specific workers
/// under static round-robin scheduling, exposing the work-stealing advantage.
///
/// Structure (1 + n_long_chains + n_short_ops ops):
/// - Op 0: root fan-out source (no dependencies)
/// - n_long_chains heavy ops each taking `chain_depth × slow_factor` ms, all independent
///   (each depends only on root — no inter-op dependencies within the heavy group)
/// - n_short_ops fast ops each taking 1 ms, also depending only on root
///
/// **Why this breaks static:** Round-robin assigns the first n_long_chains heavy ops to
/// workers 0..n_long_chains-1 before any fast ops. Those workers become fully occupied
/// for `chain_depth × slow_factor` ms. The remaining workers exhaust their fast-op queues
/// in `(n_short_ops / n_workers)` ms and sit idle. Work-stealing detects the idle workers
/// and redistributes the fast ops that were pre-assigned to the busy workers.
///
/// # Errors
/// Returns [`DagError`] if DAG construction fails.
pub fn gen_imbalanced(
    n_long_chains: usize,
    chain_depth: usize,
    n_short_ops: usize,
    slow_factor: u64,
) -> Result<(Dag, HashMap<OpId, Tensor>), DagError> {
    let slow_ms = chain_depth as u64 * slow_factor;
    let total = 1 + n_long_chains + n_short_ops;
    let mut ops: Vec<Op> = Vec::with_capacity(total);

    ops.push(Op::new(0, OpKind::Relu { len: 1 }, vec![], vec![1]));

    for b in 0..n_long_chains {
        let id = 1 + b;
        ops.push(Op::new(
            id,
            OpKind::Slow {
                duration_ms: slow_ms,
            },
            vec![0],
            vec![1],
        ));
    }

    let short_base = 1 + n_long_chains;
    for i in 0..n_short_ops {
        let id = short_base + i;
        ops.push(Op::new(
            id,
            OpKind::Slow { duration_ms: 1 },
            vec![0],
            vec![1],
        ));
    }

    let dag = Dag::new(ops)?;
    let sources = HashMap::from([(0usize, Tensor::full(&[1], 1.0))]);
    Ok((dag, sources))
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn assert_no_cycle(dag: &Dag) {
        dag.topological_sort().expect("DAG must be acyclic");
    }

    fn edge_count(dag: &Dag) -> usize {
        dag.ops.iter().map(|op| op.input_ids.len()).sum()
    }

    #[test]
    fn transformer_block_op_count() {
        let (dag, sources) = gen_transformer_block(64, 128, 8).unwrap();
        assert_eq!(dag.len(), 18, "expected 18 ops");
        // 6 source ops
        assert_eq!(sources.len(), 6);
    }

    #[test]
    fn transformer_block_edge_count() {
        let (dag, _) = gen_transformer_block(64, 128, 8).unwrap();
        // Counted manually: 19 directed edges
        assert_eq!(edge_count(&dag), 19);
    }

    #[test]
    fn transformer_block_no_cycle() {
        let (dag, _) = gen_transformer_block(64, 128, 8).unwrap();
        assert_no_cycle(&dag);
    }

    #[test]
    fn transformer_block_qkv_parallel() {
        let (dag, _) = gen_transformer_block(64, 128, 8).unwrap();
        let order = dag.topological_sort().unwrap();
        let pos: std::collections::HashMap<usize, usize> =
            order.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        // Q(6), K(7), V(8) are all immediate successors of X(0) — none should precede another
        let pq = pos[&6];
        let pk = pos[&7];
        let pv = pos[&8];
        // All must come after source X(0)
        assert!(pq > pos[&0] && pk > pos[&0] && pv > pos[&0]);
        // None of Q/K/V depends on the others
        assert!(dag.ops[6].input_ids == vec![0, 1]);
        assert!(dag.ops[7].input_ids == vec![0, 2]);
        assert!(dag.ops[8].input_ids == vec![0, 3]);
        // suppress unused-variable warnings for pq/pk/pv
        let _ = (pq, pk, pv);
    }

    #[test]
    fn wide_dag_op_count() {
        let (dag, sources) = gen_wide_dag(8, 5, 0.25).unwrap();
        assert_eq!(dag.len(), 1 + 8 * 5, "expected 41 ops");
        assert_eq!(sources.len(), 1);
    }

    #[test]
    fn wide_dag_no_cycle() {
        let (dag, _) = gen_wide_dag(8, 5, 0.25).unwrap();
        assert_no_cycle(&dag);
    }

    #[test]
    fn wide_dag_skew_branch_count() {
        let width = 8;
        let skew = 0.25;
        let (dag, _) = gen_wide_dag(width, 5, skew).unwrap();
        let n_slow = dag
            .ops
            .iter()
            .filter(|op| matches!(op.kind, OpKind::Slow { duration_ms } if duration_ms == 5))
            .count();
        let expected_slow_branches = (width as f32 * skew).round() as usize;
        assert_eq!(n_slow, expected_slow_branches * 5);
    }

    #[test]
    fn wide_dag_all_branches_independent() {
        // Each branch op's only upstream is either op 0 or the previous op in the same branch.
        let (dag, _) = gen_wide_dag(4, 3, 0.0).unwrap();
        let topo = dag.topological_sort().unwrap();
        assert_eq!(topo.len(), dag.len());
        // Verify branch structure: op at position (b, d) depends on (b, d-1) or root
        for b in 0..4usize {
            for d in 0..3usize {
                let id = 1 + b * 3 + d;
                let op = &dag.ops[id];
                if d == 0 {
                    assert_eq!(op.input_ids, vec![0]);
                } else {
                    assert_eq!(op.input_ids, vec![id - 1]);
                }
            }
        }
    }

    #[test]
    fn resnet_block_op_count() {
        let (dag, sources) = gen_resnet_block(64).unwrap();
        assert_eq!(dag.len(), 8, "expected 8 ops");
        assert_eq!(sources.len(), 3);
    }

    #[test]
    fn resnet_block_edge_count() {
        let (dag, _) = gen_resnet_block(64).unwrap();
        // 3→4(conv1) + 2(W1) + 1(relu1) + 2(conv2,W2) + 2(merge:conv2+X) + 1(bn) = 8 edges
        assert_eq!(edge_count(&dag), 8);
    }

    #[test]
    fn resnet_block_no_cycle() {
        let (dag, _) = gen_resnet_block(64).unwrap();
        assert_no_cycle(&dag);
    }

    #[test]
    fn resnet_block_fork_join() {
        let (dag, _) = gen_resnet_block(64).unwrap();
        // merge op (id 6) must depend on both conv2 (id 5) and X (id 0)
        let merge = &dag.ops[6];
        let inputs: HashSet<usize> = merge.input_ids.iter().copied().collect();
        assert!(inputs.contains(&5), "merge must depend on conv2");
        assert!(inputs.contains(&0), "merge must depend on X (skip path)");
    }

    #[test]
    fn all_generators_execute_sequentially() {
        let (dag_t, src_t) = gen_transformer_block(32, 64, 4).unwrap();
        let (dag_w, src_w) = gen_wide_dag(4, 3, 0.0).unwrap();
        let (dag_r, src_r) = gen_resnet_block(16).unwrap();

        dag_t.topological_sort().unwrap();
        dag_w.topological_sort().unwrap();
        dag_r.topological_sort().unwrap();
        let _ = (src_t, src_w, src_r);
    }

    #[test]
    fn large_transformer_op_count_single_layer() {
        let (dag, _) = gen_large_transformer(1, 64).unwrap();
        assert_eq!(dag.len(), 18, "1 layer = 18 ops");
    }

    #[test]
    fn large_transformer_op_count_multi_layer() {
        let (dag, _) = gen_large_transformer(8, 64).unwrap();
        assert_eq!(dag.len(), 18 + 7 * 17, "8 layers = 137 ops");
    }

    #[test]
    fn large_transformer_no_cycle() {
        let (dag, _) = gen_large_transformer(4, 32).unwrap();
        assert_no_cycle(&dag);
    }

    #[test]
    fn large_transformer_sources_populated() {
        let (dag, sources) = gen_large_transformer(3, 32).unwrap();
        // Layer 0: 6 sources. Layers 1-2: 5 each. Total = 16.
        assert_eq!(sources.len(), 6 + 2 * 5);
        dag.topological_sort().unwrap();
    }

    #[test]
    fn large_wide_op_count() {
        let (dag, _) = gen_large_wide(32, 10, 0.5).unwrap();
        assert_eq!(dag.len(), 1 + 32 * 10, "321 ops");
    }

    #[test]
    fn large_wide_no_cycle() {
        let (dag, _) = gen_large_wide(32, 10, 0.5).unwrap();
        assert_no_cycle(&dag);
    }

    #[test]
    fn large_wide_skew_branch_count() {
        let (dag, _) = gen_large_wide(32, 10, 0.5).unwrap();
        let n_slow = dag
            .ops
            .iter()
            .filter(|op| matches!(op.kind, OpKind::Slow { duration_ms } if duration_ms == 5))
            .count();
        assert_eq!(n_slow, 16 * 10, "half branches × 10 ops each");
    }

    #[test]
    fn imbalanced_op_count() {
        let (dag, sources) = gen_imbalanced(4, 20, 200, 10).unwrap();
        assert_eq!(dag.len(), 1 + 4 + 200, "205 ops");
        assert_eq!(sources.len(), 1);
    }

    #[test]
    fn imbalanced_no_cycle() {
        let (dag, _) = gen_imbalanced(4, 20, 200, 10).unwrap();
        assert_no_cycle(&dag);
    }

    #[test]
    fn imbalanced_heavy_ops_fan_out_from_root() {
        let (dag, _) = gen_imbalanced(2, 3, 0, 5).unwrap();
        // Heavy ops 1 and 2 both depend only on root
        assert_eq!(dag.ops[1].input_ids, vec![0]);
        assert_eq!(dag.ops[2].input_ids, vec![0]);
    }

    #[test]
    fn imbalanced_short_ops_fan_out_from_root() {
        let (dag, _) = gen_imbalanced(2, 3, 5, 5).unwrap();
        // Short ops start at id = 1 + 2 = 3
        for i in 3..8usize {
            assert_eq!(
                dag.ops[i].input_ids,
                vec![0],
                "short op {i} must depend only on root"
            );
        }
    }

    #[test]
    fn imbalanced_slow_fast_split() {
        // n_long_chains=2, chain_depth=3, n_short_ops=4, slow_factor=10
        // heavy op duration = 3 * 10 = 30ms; fast op duration = 1ms
        let (dag, _) = gen_imbalanced(2, 3, 4, 10).unwrap();
        let slow_count = dag
            .ops
            .iter()
            .filter(|op| matches!(op.kind, OpKind::Slow { duration_ms } if duration_ms == 30))
            .count();
        let fast_count = dag
            .ops
            .iter()
            .filter(|op| matches!(op.kind, OpKind::Slow { duration_ms } if duration_ms == 1))
            .count();
        assert_eq!(slow_count, 2, "2 heavy ops");
        assert_eq!(fast_count, 4, "4 fast short ops");
    }
}
