use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use ferroflow_core::{Dag, Op, OpKind, RunMetrics, SchedulerMetrics, Tensor};
use ferroflow_runtime::{SequentialExecutor, StaticScheduler, WorkStealingScheduler};

// ── CLI types ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, ValueEnum)]
enum DagKind {
    /// Linear matmul chain — no skew; all ops have equal cost.
    Uniform,
    /// Two-branch DAG with Slow ops; one branch is `skew_factor` times slower.
    Skewed,
}

/// ferroflow — distributed work-stealing tensor computation scheduler
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run all three schedulers on a given DAG topology and record RunMetrics.
    Bench {
        /// DAG topology to benchmark.
        #[arg(long, default_value = "uniform")]
        dag: DagKind,

        /// Number of parallel tokio workers (proxy for node count in local runs).
        #[arg(long, default_value = "4")]
        workers: usize,

        /// Value to record in the `nodes` field of RunMetrics (for Narval runs,
        /// set this to SLURM_NNODES).
        #[arg(long, default_value = "1")]
        nodes: u32,

        /// Number of compute ops in the DAG (must be even for skewed DAGs).
        #[arg(long, default_value = "20")]
        dag_ops: usize,

        /// Slow-branch slowdown factor (skewed DAGs only).
        #[arg(long, default_value = "5")]
        skew_factor: u64,

        /// Matrix dimension for uniform matmul chains.
        #[arg(long, default_value = "128")]
        chain_dim: usize,

        /// Append results to this JSON file (creates the file if absent).
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Print a markdown comparison table from a saved benchmark_results.json.
    Report {
        /// Path to the JSON results file.
        input: PathBuf,
    },
}

// ── DAG builders ─────────────────────────────────────────────────────────────

fn build_uniform_dag(dag_ops: usize, dim: usize) -> (Arc<Dag>, HashMap<usize, Tensor>) {
    let n_src = dag_ops + 1;
    let mut ops: Vec<Op> = (0..n_src)
        .map(|id| Op::new(id, OpKind::Matmul { m: dim, n: dim, k: dim }, vec![], vec![dim, dim]))
        .collect();

    let mut prev = 0usize;
    for i in 0..dag_ops {
        let cid = n_src + i;
        let inputs = if i == 0 { vec![0, 1] } else { vec![prev, i + 1] };
        ops.push(Op::new(cid, OpKind::Matmul { m: dim, n: dim, k: dim }, inputs, vec![dim, dim]));
        prev = cid;
    }

    let dag = Arc::new(Dag::new(ops).expect("uniform dag is acyclic"));
    let val = 1.0 / dim as f32;
    let src = (0..n_src).map(|id| (id, Tensor::full(&[dim, dim], val))).collect();
    (dag, src)
}

fn build_skewed_dag(dag_ops: usize, factor: u64) -> (Arc<Dag>, HashMap<usize, Tensor>) {
    let (dag, src) = Dag::with_skew(dag_ops, factor).expect("skewed dag is valid");
    (Arc::new(dag), src)
}

// ── Metrics helpers ───────────────────────────────────────────────────────────

fn make_run(
    scheduler: &str,
    dag_kind: &DagKind,
    nodes: u32,
    workers: u32,
    dag_size: usize,
    m: SchedulerMetrics,
) -> RunMetrics {
    RunMetrics {
        scheduler: scheduler.to_string(),
        nodes,
        workers_per_node: workers,
        dag_size: dag_size as u32,
        skew: matches!(dag_kind, DagKind::Skewed),
        metrics: m,
    }
}

// ── Bench logic ───────────────────────────────────────────────────────────────

async fn run_bench(
    dag: DagKind,
    workers: usize,
    nodes: u32,
    dag_ops: usize,
    skew_factor: u64,
    chain_dim: usize,
    output: Option<PathBuf>,
) -> Result<Vec<RunMetrics>> {
    let (arc_dag, src) = match dag {
        DagKind::Uniform => build_uniform_dag(dag_ops, chain_dim),
        DagKind::Skewed => build_skewed_dag(dag_ops, skew_factor),
    };
    let dag_size = arc_dag.len();
    let dag_label = match dag {
        DagKind::Uniform => "uniform",
        DagKind::Skewed => "skewed",
    };
    let mut results: Vec<RunMetrics> = Vec::new();

    // Sequential
    let (_, m) = SequentialExecutor::execute(&arc_dag, src.clone())
        .context("sequential executor failed")?;
    let r = make_run("sequential", &dag, nodes, 1, dag_size, m);
    println!("[bench] sequential/{dag_label}: {:.1} ms  ({:.0} ops/s)",
        r.metrics.elapsed_ms, r.metrics.throughput_ops_per_sec);
    results.push(r);

    // Static
    let sched = StaticScheduler::new(&arc_dag, workers);
    let (_, m) = sched.execute(Arc::clone(&arc_dag), src.clone()).await
        .context("static scheduler failed")?;
    let r = make_run("static", &dag, nodes, workers as u32, dag_size, m);
    println!("[bench] static/{dag_label} ({workers}w): {:.1} ms  ({:.0} ops/s)  idle={:.1}%",
        r.metrics.elapsed_ms, r.metrics.throughput_ops_per_sec,
        idle_pct(&r));
    results.push(r);

    // Work-stealing
    let sched = WorkStealingScheduler::new(workers);
    let (_, m) = sched.execute(Arc::clone(&arc_dag), src).await
        .context("work-stealing scheduler failed")?;
    let r = make_run("work-stealing", &dag, nodes, workers as u32, dag_size, m);
    println!("[bench] ws/{dag_label} ({workers}w): {:.1} ms  ({:.0} ops/s)  idle={:.1}%  steals={}/s",
        r.metrics.elapsed_ms, r.metrics.throughput_ops_per_sec,
        idle_pct(&r), r.metrics.steal_rate);
    results.push(r);

    if let Some(path) = output {
        append_json(&path, &results).with_context(|| format!("writing {}", path.display()))?;
        println!("[bench] appended {} entries → {}", results.len(), path.display());
    }

    Ok(results)
}

fn append_json(path: &PathBuf, new_entries: &[RunMetrics]) -> Result<()> {
    let mut existing: Vec<RunMetrics> = if path.exists() {
        let raw = std::fs::read_to_string(path)?;
        if raw.trim().is_empty() { vec![] } else { serde_json::from_str(&raw)? }
    } else {
        vec![]
    };
    existing.extend_from_slice(new_entries);
    let json = serde_json::to_string_pretty(&existing)?;
    std::fs::write(path, json)?;
    Ok(())
}

// ── Report ────────────────────────────────────────────────────────────────────

fn idle_pct(r: &RunMetrics) -> f64 {
    let w = r.workers_per_node.max(1) as f64;
    let total = r.metrics.elapsed_ms * w;
    if total > 0.0 { r.metrics.idle_time_ms / total * 100.0 } else { 0.0 }
}

fn print_report(entries: &[RunMetrics]) {
    println!("\n## Scheduler Comparison\n");
    println!(
        "| {:<13} | {:>5} | {:<7} | {:>14} | {:>6} | {:>10} |",
        "Scheduler", "Nodes", "DAG", "Throughput", "Idle%", "Steal Rate"
    );
    println!("|{:-<15}|{:-<7}|{:-<9}|{:-<16}|{:-<8}|{:-<12}|", "", "", "", "", "", "");
    for r in entries {
        let dag_label = if r.skew { "skewed" } else { "uniform" };
        println!(
            "| {:<13} | {:>5} | {:<7} | {:>13.0} /s | {:>5.1}% | {:>9.1}/s |",
            r.scheduler,
            r.nodes,
            dag_label,
            r.metrics.throughput_ops_per_sec,
            idle_pct(r),
            r.metrics.steal_rate,
        );
    }
    println!();
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    match args.command {
        Command::Bench { dag, workers, nodes, dag_ops, skew_factor, chain_dim, output } => {
            run_bench(dag, workers, nodes, dag_ops, skew_factor, chain_dim, output).await?;
        }
        Command::Report { input } => {
            let raw = std::fs::read_to_string(&input)
                .with_context(|| format!("reading {}", input.display()))?;
            let entries: Vec<RunMetrics> = serde_json::from_str(&raw)
                .context("parsing RunMetrics JSON")?;
            print_report(&entries);
        }
    }

    Ok(())
}
