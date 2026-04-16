/// Criterion benchmarks for the sequential executor.
///
/// Topology: a linear chain of 20 matmul ops, each multiplying the previous
/// output (128×128) by a fresh 128×128 weight matrix.
///
///   src_0 (128×128) ──▶ mm_1 ──▶ mm_2 ──▶ … ──▶ mm_20
///   src_1 (128×128) ──┘
///   src_2 (128×128) ──────────────┘
///   …
use std::collections::HashMap;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ferroflow_core::{Dag, Op, OpKind, Tensor};
use ferroflow_runtime::SequentialExecutor;

const CHAIN_LEN: usize = 20;
const DIM: usize = 128;

/// Builds a matmul chain of `chain_len` ops.
///
/// Op layout (ids are indices into the ops vec):
///   - ops 0..chain_len          : source weight matrices (no input_ids)
///   - op  chain_len             : first compute — matmul(src_0, src_1)
///   - ops chain_len+1 ..        : matmul(prev_compute, src_i+2)
///
/// Returns (dag, source_tensors).
fn build_matmul_chain(chain_len: usize, dim: usize) -> (Dag, HashMap<usize, Tensor>) {
    // chain_len weight sources + chain_len compute ops
    let n_sources = chain_len + 1; // one extra for the initial activation
    let mut ops: Vec<Op> = Vec::new();

    // Source ops: id 0 .. n_sources-1
    for id in 0..n_sources {
        ops.push(Op::new(
            id,
            OpKind::Matmul { m: dim, n: dim, k: dim },
            vec![],
            vec![dim, dim],
        ));
    }

    // Compute ops: chain of matmuls starting at id = n_sources
    // compute_0 = src_0 · src_1
    // compute_i = compute_{i-1} · src_{i+1}
    let mut prev_id = 0usize; // will be overwritten on first iteration
    for i in 0..chain_len {
        let compute_id = n_sources + i;
        let input_ids = if i == 0 {
            vec![0, 1]
        } else {
            vec![prev_id, i + 1]
        };
        ops.push(Op::new(
            compute_id,
            OpKind::Matmul { m: dim, n: dim, k: dim },
            input_ids,
            vec![dim, dim],
        ));
        prev_id = compute_id;
    }

    let dag = Dag::new(ops).expect("chain dag must be acyclic");

    // Provide identity-ish source tensors (all ones / dim).
    let val = 1.0 / dim as f32;
    let mut sources: HashMap<usize, Tensor> = HashMap::new();
    for id in 0..n_sources {
        sources.insert(id, Tensor::full(&[dim, dim], val));
    }

    (dag, sources)
}

fn bench_sequential_matmul_chain(c: &mut Criterion) {
    let (dag, sources_template) = build_matmul_chain(CHAIN_LEN, DIM);

    c.bench_function(
        &format!("sequential_{CHAIN_LEN}op_matmul_chain_{DIM}x{DIM}"),
        |b| {
            b.iter(|| {
                // Clone source tensors each iteration so allocations are included.
                let sources = sources_template.clone();
                let result = SequentialExecutor::execute(black_box(&dag), sources).unwrap();
                black_box(result)
            })
        },
    );
}

criterion_group!(benches, bench_sequential_matmul_chain);
criterion_main!(benches);
