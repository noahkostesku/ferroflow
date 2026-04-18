//! Criterion benchmarks comparing sequential, static, and work-stealing schedulers
//! across two DAG topologies.
//!
//! **Group 1 — uniform** (no skew): 20-op matmul chain, 128×128 matrices.
//! **Group 2 — skewed**: 20-op two-branch DAG built with [`Dag::with_skew`];
//!   the slow branch takes 5× longer per op than the fast branch.
//!
//! After the Criterion runs, a one-shot metrics collection pass prints a
//! markdown comparison table and saves `docs/benchmark_results.json`.

use std::collections::HashMap;
use std::sync::Arc;

use criterion::{black_box, criterion_group, Criterion};
use ferroflow_core::{Dag, Op, OpKind, RunMetrics, SchedulerMetrics, Tensor};
use ferroflow_runtime::{SequentialExecutor, StaticScheduler, WorkStealingScheduler};
use tokio::runtime::Runtime;

const CHAIN_LEN: usize = 20;
const CHAIN_DIM: usize = 128;
const N_WORKERS: usize = 4;
const SLOW_BRANCH_FACTOR: u64 = 5;

// ── DAG builders ─────────────────────────────────────────────────────────────

/// Uniform matmul chain: `chain_len` ops each computing `dim×dim` × `dim×dim`.
fn build_uniform_dag(chain_len: usize, dim: usize) -> (Arc<Dag>, HashMap<usize, Tensor>) {
    let n_sources = chain_len + 1;
    let mut ops: Vec<Op> = Vec::new();

    for id in 0..n_sources {
        ops.push(Op::new(
            id,
            OpKind::Matmul {
                m: dim,
                n: dim,
                k: dim,
            },
            vec![],
            vec![dim, dim],
        ));
    }

    let mut prev_id = 0usize;
    for i in 0..chain_len {
        let compute_id = n_sources + i;
        let input_ids = if i == 0 {
            vec![0, 1]
        } else {
            vec![prev_id, i + 1]
        };
        ops.push(Op::new(
            compute_id,
            OpKind::Matmul {
                m: dim,
                n: dim,
                k: dim,
            },
            input_ids,
            vec![dim, dim],
        ));
        prev_id = compute_id;
    }

    let dag = Arc::new(Dag::new(ops).expect("uniform dag is acyclic"));
    let val = 1.0 / dim as f32;
    let sources = (0..n_sources)
        .map(|id| (id, Tensor::full(&[dim, dim], val)))
        .collect();
    (dag, sources)
}

/// Two-branch skewed DAG via [`Dag::with_skew`].
fn build_skewed_dag(n_ops: usize, factor: u64) -> (Arc<Dag>, HashMap<usize, Tensor>) {
    let (dag, sources) = Dag::with_skew(n_ops, factor).expect("skewed dag is valid");
    (Arc::new(dag), sources)
}

// ── Metrics collection ───────────────────────────────────────────────────────

fn make_run_metrics(
    scheduler: &str,
    skew: bool,
    dag_size: usize,
    workers: u32,
    m: SchedulerMetrics,
) -> RunMetrics {
    RunMetrics {
        scheduler: scheduler.to_string(),
        nodes: 1,
        workers_per_node: workers,
        dag_size: dag_size as u32,
        skew,
        metrics: m,
    }
}

/// Runs all six variants once and returns their [`RunMetrics`].
fn collect_all_metrics() -> Vec<RunMetrics> {
    let rt = Runtime::new().expect("tokio runtime");
    let mut results = Vec::with_capacity(6);

    // ── uniform ──────────────────────────────────────────────────────────────

    let (dag, src) = build_uniform_dag(CHAIN_LEN, CHAIN_DIM);
    let (_, m) = SequentialExecutor::execute(&dag, src).unwrap();
    results.push(make_run_metrics("sequential", false, dag.len(), 1, m));

    let (dag, src) = build_uniform_dag(CHAIN_LEN, CHAIN_DIM);
    let sched = StaticScheduler::new(&dag, N_WORKERS);
    let (_, m) = rt.block_on(sched.execute(Arc::clone(&dag), src)).unwrap();
    results.push(make_run_metrics(
        "static",
        false,
        dag.len(),
        N_WORKERS as u32,
        m,
    ));

    let (dag, src) = build_uniform_dag(CHAIN_LEN, CHAIN_DIM);
    let sched = WorkStealingScheduler::new(N_WORKERS);
    let (_, m) = rt.block_on(sched.execute(Arc::clone(&dag), src)).unwrap();
    results.push(make_run_metrics(
        "work-stealing",
        false,
        dag.len(),
        N_WORKERS as u32,
        m,
    ));

    // ── skewed ───────────────────────────────────────────────────────────────

    let (dag, src) = build_skewed_dag(CHAIN_LEN, SLOW_BRANCH_FACTOR);
    let (_, m) = SequentialExecutor::execute(&dag, src).unwrap();
    results.push(make_run_metrics("sequential", true, dag.len(), 1, m));

    let (dag, src) = build_skewed_dag(CHAIN_LEN, SLOW_BRANCH_FACTOR);
    let sched = StaticScheduler::new(&dag, N_WORKERS);
    let (_, m) = rt.block_on(sched.execute(Arc::clone(&dag), src)).unwrap();
    results.push(make_run_metrics(
        "static",
        true,
        dag.len(),
        N_WORKERS as u32,
        m,
    ));

    let (dag, src) = build_skewed_dag(CHAIN_LEN, SLOW_BRANCH_FACTOR);
    let sched = WorkStealingScheduler::new(N_WORKERS);
    let (_, m) = rt.block_on(sched.execute(Arc::clone(&dag), src)).unwrap();
    results.push(make_run_metrics(
        "work-stealing",
        true,
        dag.len(),
        N_WORKERS as u32,
        m,
    ));

    results
}

// ── Output helpers ────────────────────────────────────────────────────────────

fn idle_pct(r: &RunMetrics) -> f64 {
    let workers = r.workers_per_node.max(1) as f64;
    let total_worker_ms = r.metrics.elapsed_ms * workers;
    if total_worker_ms > 0.0 {
        r.metrics.idle_time_ms / total_worker_ms * 100.0
    } else {
        0.0
    }
}

fn print_markdown_table(results: &[RunMetrics]) {
    println!("\n## Scheduler Comparison\n");
    println!(
        "| {:<13} | {:<7} | {:>14} | {:>6} | {:>10} |",
        "Scheduler", "DAG", "Throughput", "Idle%", "Steal Rate"
    );
    println!(
        "|{:-<15}|{:-<9}|{:-<16}|{:-<8}|{:-<12}|",
        "", "", "", "", ""
    );
    for r in results {
        let dag_label = if r.skew { "skewed" } else { "uniform" };
        let throughput = format!("{:.0} ops/s", r.metrics.throughput_ops_per_sec);
        let idle = format!("{:.1}%", idle_pct(r));
        let steal = format!("{:.1}/s", r.metrics.steal_rate);
        println!(
            "| {:<13} | {:<7} | {:>14} | {:>6} | {:>10} |",
            r.scheduler, dag_label, throughput, idle, steal
        );
    }
    println!();
}

fn save_json(results: &[RunMetrics]) {
    let json = serde_json::to_string_pretty(results).expect("serialize results");
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/benchmark_results.json");
    if let Err(e) = std::fs::write(&path, &json) {
        eprintln!("warning: could not save benchmark_results.json: {e}");
    } else {
        println!("Saved {}", path.display());
    }
}

// ── Criterion bench functions ─────────────────────────────────────────────────

fn bench_sequential_uniform(c: &mut Criterion) {
    let (dag, sources) = build_uniform_dag(CHAIN_LEN, CHAIN_DIM);
    c.bench_function(
        &format!("sequential_uniform_{CHAIN_LEN}op_{CHAIN_DIM}x{CHAIN_DIM}"),
        |b| {
            b.iter(|| {
                black_box(
                    SequentialExecutor::execute(black_box(&dag), sources.clone())
                        .unwrap()
                        .0,
                )
            })
        },
    );
}

fn bench_static_uniform(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(N_WORKERS)
        .enable_time()
        .build()
        .unwrap();
    let (dag, sources) = build_uniform_dag(CHAIN_LEN, CHAIN_DIM);
    let sched = StaticScheduler::new(&dag, N_WORKERS);
    c.bench_function(
        &format!("static_{N_WORKERS}w_uniform_{CHAIN_LEN}op_{CHAIN_DIM}x{CHAIN_DIM}"),
        |b| {
            b.iter(|| {
                black_box(
                    rt.block_on(sched.execute(Arc::clone(&dag), sources.clone()))
                        .unwrap()
                        .0,
                )
            })
        },
    );
}

fn bench_workstealing_uniform(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(N_WORKERS)
        .enable_time()
        .build()
        .unwrap();
    let (dag, sources) = build_uniform_dag(CHAIN_LEN, CHAIN_DIM);
    let sched = WorkStealingScheduler::new(N_WORKERS);
    c.bench_function(
        &format!("ws_{N_WORKERS}w_uniform_{CHAIN_LEN}op_{CHAIN_DIM}x{CHAIN_DIM}"),
        |b| {
            b.iter(|| {
                black_box(
                    rt.block_on(sched.execute(Arc::clone(&dag), sources.clone()))
                        .unwrap()
                        .0,
                )
            })
        },
    );
}

fn bench_sequential_skewed(c: &mut Criterion) {
    let (dag, sources) = build_skewed_dag(CHAIN_LEN, SLOW_BRANCH_FACTOR);
    c.bench_function("sequential_skewed_20op_factor5", |b| {
        b.iter(|| {
            black_box(
                SequentialExecutor::execute(black_box(&dag), sources.clone())
                    .unwrap()
                    .0,
            )
        })
    });
}

fn bench_static_skewed(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(N_WORKERS)
        .enable_time()
        .build()
        .unwrap();
    let (dag, sources) = build_skewed_dag(CHAIN_LEN, SLOW_BRANCH_FACTOR);
    let sched = StaticScheduler::new(&dag, N_WORKERS);
    c.bench_function(&format!("static_{N_WORKERS}w_skewed_20op_factor5"), |b| {
        b.iter(|| {
            black_box(
                rt.block_on(sched.execute(Arc::clone(&dag), sources.clone()))
                    .unwrap()
                    .0,
            )
        })
    });
}

fn bench_workstealing_skewed(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(N_WORKERS)
        .enable_time()
        .build()
        .unwrap();
    let (dag, sources) = build_skewed_dag(CHAIN_LEN, SLOW_BRANCH_FACTOR);
    let sched = WorkStealingScheduler::new(N_WORKERS);
    c.bench_function(&format!("ws_{N_WORKERS}w_skewed_20op_factor5"), |b| {
        b.iter(|| {
            black_box(
                rt.block_on(sched.execute(Arc::clone(&dag), sources.clone()))
                    .unwrap()
                    .0,
            )
        })
    });
}

// ── Groups ────────────────────────────────────────────────────────────────────

criterion_group!(
    uniform_benches,
    bench_sequential_uniform,
    bench_static_uniform,
    bench_workstealing_uniform,
);

criterion_group!(
    skewed_benches,
    bench_sequential_skewed,
    bench_static_skewed,
    bench_workstealing_skewed,
);

// ── Custom main ───────────────────────────────────────────────────────────────

fn main() {
    // Run Criterion timing (each group creates its own Criterion instance).
    uniform_benches();
    skewed_benches();

    // One-shot metrics pass: collect RunMetrics for the comparison table.
    eprintln!("\nCollecting one-shot metrics for comparison table...");
    let results = collect_all_metrics();

    print_markdown_table(&results);
    save_json(&results);
}
