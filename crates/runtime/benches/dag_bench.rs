//! Criterion benchmarks comparing sequential, static, and work-stealing schedulers.
//!
//! Topology A — balanced matmul chain:
//!   src_0 (128×128) ──▶ mm_1 ──▶ mm_2 ──▶ … ──▶ mm_20
//!   src_1 (128×128) ──┘
//!   …
//!
//! Topology B — skewed fan-out (8 heavy + 8 light independent ops):
//!   src_0, src_1 (192×192) ──▶ heavy_ops[0..7]   (192×192 matmul each)
//!   src_2, src_3  (48×48)  ──▶ light_ops[0..7]   (48×48  matmul each)
//!
//!   With 4 workers and round-robin assignment, heavy ops land exclusively on
//!   workers 0 and 1 (op_id % 4 ∈ {0,1}), leaving workers 2 and 3 with only
//!   light ops.  Work-stealing redistributes the heavy ops across all four
//!   workers once the fast workers become idle.

use std::collections::HashMap;
use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ferroflow_core::{Dag, Op, OpKind, Tensor};
use ferroflow_runtime::{SequentialExecutor, StaticScheduler, WorkStealingScheduler};

const CHAIN_LEN: usize = 20;
const CHAIN_DIM: usize = 128;
const N_WORKERS: usize = 4;

// ── DAG builders ─────────────────────────────────────────────────────────────

/// Builds a linear matmul chain of `chain_len` ops (each `dim×dim`).
///
/// Returns `(dag, source_tensors)`.
fn build_matmul_chain(chain_len: usize, dim: usize) -> (Arc<Dag>, HashMap<usize, Tensor>) {
    let n_sources = chain_len + 1;
    let mut ops: Vec<Op> = Vec::new();

    for id in 0..n_sources {
        ops.push(Op::new(
            id,
            OpKind::Matmul { m: dim, n: dim, k: dim },
            vec![],
            vec![dim, dim],
        ));
    }

    let mut prev_id = 0usize;
    for i in 0..chain_len {
        let compute_id = n_sources + i;
        let input_ids = if i == 0 { vec![0, 1] } else { vec![prev_id, i + 1] };
        ops.push(Op::new(
            compute_id,
            OpKind::Matmul { m: dim, n: dim, k: dim },
            input_ids,
            vec![dim, dim],
        ));
        prev_id = compute_id;
    }

    let dag = Arc::new(Dag::new(ops).expect("chain dag is acyclic"));
    let val = 1.0 / dim as f32;
    let sources = (0..n_sources).map(|id| (id, Tensor::full(&[dim, dim], val))).collect();
    (dag, sources)
}

/// Builds a skewed fan-out DAG designed to expose static-scheduler imbalance.
///
/// Structure (total 20 ops):
/// - ops 0..3 : source tensors (4 sources)
/// - ops 4..19: 16 independent compute ops, all depending only on sources
///   - `op_id % 4 ∈ {0, 1}` (ids 4,5,8,9,12,13,16,17) → HEAVY: 192×192 matmul
///   - `op_id % 4 ∈ {2, 3}` (ids 6,7,10,11,14,15,18,19) → LIGHT: 48×48  matmul
///
/// With n_workers=4 and round-robin: workers 0 and 1 each receive 4 heavy ops,
/// workers 2 and 3 receive 4 light ops each.  Work-stealing allows workers 2
/// and 3 to steal heavy ops once their queues are drained.
fn build_skewed_dag() -> (Arc<Dag>, HashMap<usize, Tensor>) {
    const HEAVY: usize = 192;
    const LIGHT: usize = 48;

    // 4 sources: 0,1 feed heavy ops; 2,3 feed light ops.
    let mut ops = vec![
        Op::new(0, OpKind::Matmul { m: HEAVY, n: HEAVY, k: HEAVY }, vec![], vec![HEAVY, HEAVY]),
        Op::new(1, OpKind::Matmul { m: HEAVY, n: HEAVY, k: HEAVY }, vec![], vec![HEAVY, HEAVY]),
        Op::new(2, OpKind::Matmul { m: LIGHT, n: LIGHT, k: LIGHT }, vec![], vec![LIGHT, LIGHT]),
        Op::new(3, OpKind::Matmul { m: LIGHT, n: LIGHT, k: LIGHT }, vec![], vec![LIGHT, LIGHT]),
    ];

    // 16 compute ops (ids 4..19).
    for id in 4..20usize {
        let is_heavy = id % 4 < 2; // ids ≡ 0,1 mod 4
        if is_heavy {
            ops.push(Op::new(
                id,
                OpKind::Matmul { m: HEAVY, n: HEAVY, k: HEAVY },
                vec![0, 1],
                vec![HEAVY, HEAVY],
            ));
        } else {
            ops.push(Op::new(
                id,
                OpKind::Matmul { m: LIGHT, n: LIGHT, k: LIGHT },
                vec![2, 3],
                vec![LIGHT, LIGHT],
            ));
        }
    }

    let dag = Arc::new(Dag::new(ops).expect("skewed dag is acyclic"));

    let heavy_val = 1.0 / HEAVY as f32;
    let light_val = 1.0 / LIGHT as f32;
    let sources = HashMap::from([
        (0, Tensor::full(&[HEAVY, HEAVY], heavy_val)),
        (1, Tensor::full(&[HEAVY, HEAVY], heavy_val)),
        (2, Tensor::full(&[LIGHT, LIGHT], light_val)),
        (3, Tensor::full(&[LIGHT, LIGHT], light_val)),
    ]);

    (dag, sources)
}

// ── Benchmarks ───────────────────────────────────────────────────────────────

fn bench_sequential_chain(c: &mut Criterion) {
    let (dag, sources) = build_matmul_chain(CHAIN_LEN, CHAIN_DIM);

    c.bench_function(
        &format!("sequential_{CHAIN_LEN}op_matmul_chain_{CHAIN_DIM}x{CHAIN_DIM}"),
        |b| {
            b.iter(|| {
                let s = sources.clone();
                black_box(SequentialExecutor::execute(black_box(&dag), s).unwrap())
            })
        },
    );
}

fn bench_static_chain(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(N_WORKERS)
        .enable_time()
        .build()
        .unwrap();
    let (dag, sources) = build_matmul_chain(CHAIN_LEN, CHAIN_DIM);
    let sched = StaticScheduler::new(&dag, N_WORKERS);

    c.bench_function(
        &format!("static_{N_WORKERS}w_{CHAIN_LEN}op_matmul_chain_{CHAIN_DIM}x{CHAIN_DIM}"),
        |b| {
            b.iter(|| {
                let s = sources.clone();
                black_box(
                    rt.block_on(sched.execute(Arc::clone(&dag), s)).unwrap(),
                )
            })
        },
    );
}

fn bench_ws_chain(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(N_WORKERS)
        .enable_time()
        .build()
        .unwrap();
    let (dag, sources) = build_matmul_chain(CHAIN_LEN, CHAIN_DIM);
    let sched = WorkStealingScheduler::new(N_WORKERS);

    c.bench_function(
        &format!("ws_{N_WORKERS}w_{CHAIN_LEN}op_matmul_chain_{CHAIN_DIM}x{CHAIN_DIM}"),
        |b| {
            b.iter(|| {
                let s = sources.clone();
                black_box(
                    rt.block_on(sched.execute(Arc::clone(&dag), s)).unwrap(),
                )
            })
        },
    );
}

fn bench_static_skewed(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(N_WORKERS)
        .enable_time()
        .build()
        .unwrap();
    let (dag, sources) = build_skewed_dag();
    let sched = StaticScheduler::new(&dag, N_WORKERS);

    c.bench_function("static_4w_skewed_dag", |b| {
        b.iter(|| {
            let s = sources.clone();
            black_box(rt.block_on(sched.execute(Arc::clone(&dag), s)).unwrap())
        })
    });
}

fn bench_ws_skewed(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(N_WORKERS)
        .enable_time()
        .build()
        .unwrap();
    let (dag, sources) = build_skewed_dag();
    let sched = WorkStealingScheduler::new(N_WORKERS);

    c.bench_function("ws_4w_skewed_dag", |b| {
        b.iter(|| {
            let s = sources.clone();
            black_box(rt.block_on(sched.execute(Arc::clone(&dag), s)).unwrap())
        })
    });
}

criterion_group!(
    benches,
    bench_sequential_chain,
    bench_static_chain,
    bench_ws_chain,
    bench_static_skewed,
    bench_ws_skewed,
);
criterion_main!(benches);
