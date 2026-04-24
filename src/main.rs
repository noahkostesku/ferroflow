use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use ferroflow_core::{
    gen_imbalanced, gen_large_transformer, gen_large_wide, gen_resnet_block, gen_transformer_block,
    gen_wide_dag, gen_xlarge_transformer, gen_xlarge_wide, Dag, Op, OpKind, RunMetrics,
    SchedulerMetrics, Tensor,
};
use ferroflow_onnx::{dag_summary, load_model};
use ferroflow_runtime::{SequentialExecutor, StaticScheduler, WorkStealingScheduler};

// ── CLI types ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, ValueEnum)]
enum LocalDagKind {
    Uniform,
    Skewed,
}

/// Synthetic DAG topology for `run` and `info` subcommands.
#[derive(Debug, Clone, ValueEnum)]
enum SyntheticDag {
    /// Transformer attention + FFN block (~18 ops, 3-way QKV parallelism).
    Transformer,
    /// Wide fan-out DAG with configurable branch count, depth, and skew fraction.
    Wide,
    /// ResNet residual block (~8 ops, fork-join pattern).
    Resnet,
    /// Stacked multi-layer transformer (layers=8, d_model=512 → 137 ops).
    #[value(name = "large-transformer")]
    LargeTransformer,
    /// Large wide fan-out DAG (width=32, depth=10, skew=0.5 → 321 ops).
    #[value(name = "large-wide")]
    LargeWide,
    /// Imbalanced DAG: sequential slow chains + independent fast ops (exposes WS advantage over static).
    Imbalanced,
    /// Extra-large wide fan-out DAG (≥641 ops at width=64 depth=10 skew=0.5).
    #[value(name = "xlarge-wide")]
    XlargeWide,
    /// Extra-large stacked transformer (≥545 ops at layers=32 d_model=512 n_heads=8).
    #[value(name = "xlarge-transformer")]
    XlargeTransformer,
}

#[derive(Debug, Clone, ValueEnum)]
enum OnnxScheduler {
    Sequential,
    Static,
    #[value(name = "work-stealing")]
    WorkStealing,
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
    /// Run all three local schedulers on a DAG and record RunMetrics.
    ///
    /// Runs sequential, static, and work-stealing in-process and appends results
    /// to the output file (JSON array).
    Bench {
        #[arg(long, default_value = "uniform")]
        dag: LocalDagKind,
        #[arg(long, default_value = "4")]
        workers: usize,
        #[arg(long, default_value = "1")]
        nodes: u32,
        #[arg(long, default_value = "20")]
        dag_ops: usize,
        #[arg(long, default_value = "5")]
        skew_factor: u64,
        #[arg(long, default_value = "128")]
        chain_dim: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },

    /// Print a markdown comparison table from a saved RunMetrics JSON file.
    Report { input: PathBuf },

    /// Print the DAG summary (op count, edge count, type breakdown).
    ///
    /// Supply exactly one of `--model` (ONNX file) or `--dag` (synthetic generator).
    Info {
        /// Path to an ONNX model file.
        #[arg(long, conflicts_with = "dag")]
        model: Option<PathBuf>,
        /// Synthetic DAG topology.
        #[arg(long, conflicts_with = "model")]
        dag: Option<SyntheticDag>,
        // Transformer params
        #[arg(long, default_value = "64")]
        seq_len: usize,
        #[arg(long, default_value = "256")]
        d_model: usize,
        #[arg(long, default_value = "8")]
        n_heads: usize,
        // Wide dag params
        #[arg(long, default_value = "8")]
        width: usize,
        #[arg(long, default_value = "5")]
        depth: usize,
        #[arg(long, default_value = "0.25")]
        skew: f32,
        // ResNet params
        #[arg(long, default_value = "64")]
        channels: usize,
        // Large-transformer params
        #[arg(long, default_value = "8")]
        layers: usize,
        // Imbalanced params
        #[arg(long, default_value = "4")]
        n_long_chains: usize,
        #[arg(long, default_value = "20")]
        chain_depth: usize,
        #[arg(long, default_value = "200")]
        n_short_ops: usize,
        #[arg(long, default_value = "10")]
        slow_factor: u64,
    },

    /// Execute a DAG with the selected scheduler and print RunMetrics.
    ///
    /// Supply exactly one of `--model` (ONNX file) or `--dag` (synthetic generator).
    Run {
        /// Path to an ONNX model file.
        #[arg(long, conflicts_with = "dag")]
        model: Option<PathBuf>,
        /// Synthetic DAG topology.
        #[arg(long, conflicts_with = "model")]
        dag: Option<SyntheticDag>,
        /// Number of worker threads.
        #[arg(long, default_value = "4")]
        workers: usize,
        /// Scheduling strategy.
        #[arg(long, default_value = "work-stealing")]
        scheduler: OnnxScheduler,
        /// Minimum victim queue depth before work-stealing is allowed (work-stealing only).
        #[arg(long, default_value = "2")]
        steal_threshold: usize,
        // Transformer params
        #[arg(long, default_value = "64")]
        seq_len: usize,
        #[arg(long, default_value = "256")]
        d_model: usize,
        #[arg(long, default_value = "8")]
        n_heads: usize,
        // Wide dag params
        #[arg(long, default_value = "8")]
        width: usize,
        #[arg(long, default_value = "5")]
        depth: usize,
        #[arg(long, default_value = "0.25")]
        skew: f32,
        // ResNet params
        #[arg(long, default_value = "64")]
        channels: usize,
        // Large-transformer params
        #[arg(long, default_value = "8")]
        layers: usize,
        // Imbalanced params
        #[arg(long, default_value = "4")]
        n_long_chains: usize,
        #[arg(long, default_value = "20")]
        chain_depth: usize,
        #[arg(long, default_value = "200")]
        n_short_ops: usize,
        #[arg(long, default_value = "10")]
        slow_factor: u64,
    },

    /// Run the MPI distributed scheduler across all MPI ranks.
    ///
    /// All ranks must call this simultaneously (typical SPMD pattern).
    /// Rank 0 prints results and optionally appends to the output JSON file;
    /// worker ranks exit silently after receiving Shutdown.
    ///
    /// Build with `--features distributed` to enable this subcommand.
    #[cfg(feature = "distributed")]
    MpiBench {
        /// DAG topology: uniform (all ops equal cost) or skewed (half slow, half fast).
        #[arg(long, default_value = "uniform")]
        dag: MpiDagKind,

        /// Scheduling mode: static pre-assignment or demand-driven work-stealing.
        #[arg(long, default_value = "work-stealing")]
        scheduler: MpiSchedulerKind,

        /// Total compute ops in the DAG (must be even for skewed).
        #[arg(long, default_value = "20")]
        n_ops: usize,

        /// Wall-clock sleep per op for uniform DAGs, and for the fast branch of
        /// skewed DAGs (milliseconds).
        #[arg(long, default_value = "50")]
        op_duration_ms: u64,

        /// Slow-branch multiplier for skewed DAGs (slow_ms = op_duration_ms * factor).
        #[arg(long, default_value = "5")]
        skew_factor: u64,

        /// Value written to the `nodes` field of RunMetrics.
        /// Set to $SLURM_NNODES in the batch script.
        #[arg(long, default_value = "1")]
        nodes: u32,

        /// Append result to this JSON file (rank 0 only).
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

#[cfg(feature = "distributed")]
#[derive(Debug, Clone, ValueEnum)]
enum MpiDagKind {
    /// All compute ops have the same cost (Slow(op_duration_ms)).
    Uniform,
    /// Alternating slow/fast ops: odd IDs get Slow(op_duration_ms * skew_factor),
    /// even IDs get Slow(op_duration_ms).  With static scheduling and 2 workers,
    /// all slow ops land on rank 1 and all fast ops on rank 2.
    Skewed,
}

#[cfg(feature = "distributed")]
#[derive(Debug, Clone, ValueEnum)]
enum MpiSchedulerKind {
    Static,
    #[value(name = "work-stealing")]
    WorkStealing,
}

/// Groups all synthetic DAG topology parameters to avoid long argument lists.
struct SyntheticDagParams {
    seq_len: usize,
    d_model: usize,
    n_heads: usize,
    width: usize,
    depth: usize,
    skew: f32,
    channels: usize,
    layers: usize,
    n_long_chains: usize,
    chain_depth: usize,
    n_short_ops: usize,
    slow_factor: u64,
}

// ── local DAG builders ────────────────────────────────────────────────────────

fn build_uniform_dag(dag_ops: usize, dim: usize) -> (Arc<Dag>, HashMap<usize, Tensor>) {
    let n_src = dag_ops + 1;
    let mut ops: Vec<Op> = (0..n_src)
        .map(|id| {
            Op::new(
                id,
                OpKind::Matmul {
                    m: dim,
                    n: dim,
                    k: dim,
                },
                vec![],
                vec![dim, dim],
            )
        })
        .collect();
    let mut prev = 0usize;
    for i in 0..dag_ops {
        let cid = n_src + i;
        let inputs = if i == 0 {
            vec![0, 1]
        } else {
            vec![prev, i + 1]
        };
        ops.push(Op::new(
            cid,
            OpKind::Matmul {
                m: dim,
                n: dim,
                k: dim,
            },
            inputs,
            vec![dim, dim],
        ));
        prev = cid;
    }
    let dag = Arc::new(Dag::new(ops).expect("uniform dag is valid"));
    let val = 1.0 / dim as f32;
    let src = (0..n_src)
        .map(|id| (id, Tensor::full(&[dim, dim], val)))
        .collect();
    (dag, src)
}

fn build_local_skewed_dag(dag_ops: usize, factor: u64) -> (Arc<Dag>, HashMap<usize, Tensor>) {
    let (dag, src) = Dag::with_skew(dag_ops, factor).expect("skewed dag is valid");
    (Arc::new(dag), src)
}

fn build_synthetic_dag(
    kind: &SyntheticDag,
    p: &SyntheticDagParams,
) -> anyhow::Result<(Arc<Dag>, std::collections::HashMap<usize, Tensor>)> {
    let (dag, src) = match kind {
        SyntheticDag::Transformer => gen_transformer_block(p.seq_len, p.d_model, p.n_heads)?,
        SyntheticDag::Wide => gen_wide_dag(p.width, p.depth, p.skew)?,
        SyntheticDag::Resnet => gen_resnet_block(p.channels)?,
        SyntheticDag::LargeTransformer => gen_large_transformer(p.layers, p.d_model)?,
        SyntheticDag::LargeWide => gen_large_wide(p.width, p.depth, p.skew)?,
        SyntheticDag::Imbalanced => {
            gen_imbalanced(p.n_long_chains, p.chain_depth, p.n_short_ops, p.slow_factor)?
        }
        SyntheticDag::XlargeWide => gen_xlarge_wide(p.width, p.depth, p.skew)?,
        SyntheticDag::XlargeTransformer => gen_xlarge_transformer(p.layers, p.d_model, p.n_heads)?,
    };
    Ok((Arc::new(dag), src))
}

fn synthetic_dag_summary(dag: &Dag) -> String {
    let n_ops = dag.ops.len();
    let n_edges: usize = dag.ops.iter().map(|op| op.input_ids.len()).sum();
    let n_sources = dag.ops.iter().filter(|op| op.input_ids.is_empty()).count();
    let n_compute = n_ops - n_sources;

    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for op in &dag.ops {
        if !op.input_ids.is_empty() {
            let label = match &op.kind {
                OpKind::Matmul { .. } => "matmul",
                OpKind::Relu { .. } => "relu",
                OpKind::LayerNorm { .. } => "layer_norm",
                OpKind::Reduce { .. } => "reduce",
                OpKind::Slow { .. } => "slow",
            };
            *counts.entry(label).or_default() += 1;
        }
    }

    let mut breakdown: Vec<String> = counts
        .iter()
        .map(|(&k, &v)| format!("  {k}: {v}"))
        .collect();
    breakdown.sort();
    format!(
        "{n_ops} ops ({n_sources} sources, {n_compute} compute), {n_edges} edges\n{}",
        breakdown.join("\n")
    )
}

// ── MPI DAG builders ──────────────────────────────────────────────────────────

/// Uniform MPI benchmark DAG: one source op + `n_ops` independent `Slow(duration_ms)` ops.
///
/// All compute ops fan out from op 0 (the source).  With static scheduling and
/// `n_workers` workers, each worker gets `n_ops / n_workers` ops — perfectly
/// balanced.
#[cfg(feature = "distributed")]
fn build_mpi_uniform_dag(n_ops: usize, duration_ms: u64) -> (Arc<Dag>, HashMap<usize, Tensor>) {
    let mut ops = vec![Op::new(0, OpKind::Relu { len: 1 }, vec![], vec![1])];
    for id in 1..=n_ops {
        ops.push(Op::new(id, OpKind::Slow { duration_ms }, vec![0], vec![1]));
    }
    let dag = Arc::new(Dag::new(ops).expect("mpi uniform dag is valid"));
    let sources = HashMap::from([(0usize, Tensor::full(&[1], 1.0))]);
    (dag, sources)
}

/// Skewed MPI benchmark DAG: one source + `n_ops` independent ops where
/// **odd-ID ops are slow** (`duration_ms * skew_factor`) and **even-ID ops are fast**
/// (`duration_ms`).
///
/// With static scheduling by `(op_id − 1) % n_workers`:
/// - `n_workers = 2`: all slow ops → rank 1, all fast ops → rank 2.
/// - `n_workers = 4`: ranks 1 and 3 are slow, ranks 2 and 4 are fast.
///
/// Work-stealing should recover the imbalance by redistributing ops from the
/// coordinator's global ready queue as workers become idle.
#[cfg(feature = "distributed")]
fn build_mpi_skewed_dag(
    n_ops: usize,
    duration_ms: u64,
    skew_factor: u64,
) -> (Arc<Dag>, HashMap<usize, Tensor>) {
    let slow_ms = duration_ms * skew_factor;
    let mut ops = vec![Op::new(0, OpKind::Relu { len: 1 }, vec![], vec![1])];
    for id in 1..=n_ops {
        let ms = if id % 2 == 1 { slow_ms } else { duration_ms };
        ops.push(Op::new(
            id,
            OpKind::Slow { duration_ms: ms },
            vec![0],
            vec![1],
        ));
    }
    let dag = Arc::new(Dag::new(ops).expect("mpi skewed dag is valid"));
    let sources = HashMap::from([(0usize, Tensor::full(&[1], 1.0))]);
    (dag, sources)
}

// ── shared output helpers ─────────────────────────────────────────────────────

fn make_run_metrics(
    scheduler: &str,
    skew: bool,
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
        skew,
        metrics: m,
    }
}

fn idle_pct(r: &RunMetrics) -> f64 {
    let w = r.workers_per_node.max(1) as f64;
    let total = r.metrics.elapsed_ms * w;
    if total > 0.0 {
        r.metrics.idle_time_ms / total * 100.0
    } else {
        0.0
    }
}

fn print_report(entries: &[RunMetrics]) {
    println!("\n## Scheduler Comparison\n");
    println!(
        "| {:<16} | {:>5} | {:<7} | {:>14} | {:>6} | {:>10} |",
        "Scheduler", "Nodes", "DAG", "Throughput", "Idle%", "Steal Rate"
    );
    println!(
        "|{:-<18}|{:-<7}|{:-<9}|{:-<16}|{:-<8}|{:-<12}|",
        "", "", "", "", "", ""
    );
    for r in entries {
        println!(
            "| {:<16} | {:>5} | {:<7} | {:>13.0} /s | {:>5.1}% | {:>9.1}/s |",
            r.scheduler,
            r.nodes,
            if r.skew { "skewed" } else { "uniform" },
            r.metrics.throughput_ops_per_sec,
            idle_pct(r),
            r.metrics.steal_rate,
        );
    }
    println!();
}

fn append_json(path: &PathBuf, new_entries: &[RunMetrics]) -> Result<()> {
    let mut existing: Vec<RunMetrics> = if path.exists() {
        let raw = std::fs::read_to_string(path)?;
        if raw.trim().is_empty() {
            vec![]
        } else {
            serde_json::from_str(&raw)?
        }
    } else {
        vec![]
    };
    existing.extend_from_slice(new_entries);
    std::fs::write(path, serde_json::to_string_pretty(&existing)?)?;
    Ok(())
}

// ── subcommand handlers ───────────────────────────────────────────────────────

async fn run_local_bench(
    dag: LocalDagKind,
    workers: usize,
    nodes: u32,
    dag_ops: usize,
    skew_factor: u64,
    chain_dim: usize,
    output: Option<PathBuf>,
) -> Result<()> {
    let (arc_dag, src) = match dag {
        LocalDagKind::Uniform => build_uniform_dag(dag_ops, chain_dim),
        LocalDagKind::Skewed => build_local_skewed_dag(dag_ops, skew_factor),
    };
    let dag_size = arc_dag.len();
    let skew = matches!(dag, LocalDagKind::Skewed);
    let dag_label = if skew { "skewed" } else { "uniform" };
    let mut results: Vec<RunMetrics> = Vec::new();

    let (_, m) =
        SequentialExecutor::execute(&arc_dag, src.clone()).context("sequential executor failed")?;
    let r = make_run_metrics("sequential", skew, nodes, 1, dag_size, m);
    println!(
        "[bench] sequential/{dag_label}: {:.1} ms  ({:.0} ops/s)",
        r.metrics.elapsed_ms, r.metrics.throughput_ops_per_sec
    );
    results.push(r);

    let sched = StaticScheduler::new(&arc_dag, workers);
    let (_, m) = sched
        .execute(Arc::clone(&arc_dag), src.clone())
        .await
        .context("static scheduler failed")?;
    let r = make_run_metrics("static", skew, nodes, workers as u32, dag_size, m);
    println!(
        "[bench] static/{dag_label} ({workers}w): {:.1} ms  ({:.0} ops/s)  idle={:.1}%",
        r.metrics.elapsed_ms,
        r.metrics.throughput_ops_per_sec,
        idle_pct(&r)
    );
    results.push(r);

    let sched = WorkStealingScheduler::new(workers);
    let (_, m) = sched
        .execute(Arc::clone(&arc_dag), src)
        .await
        .context("work-stealing failed")?;
    let r = make_run_metrics("work-stealing", skew, nodes, workers as u32, dag_size, m);
    println!(
        "[bench] ws/{dag_label} ({workers}w): {:.1} ms  ({:.0} ops/s)  idle={:.1}%  steals={:.1}/s",
        r.metrics.elapsed_ms,
        r.metrics.throughput_ops_per_sec,
        idle_pct(&r),
        r.metrics.steal_rate
    );
    results.push(r);

    if let Some(path) = output {
        append_json(&path, &results).with_context(|| format!("writing {}", path.display()))?;
        println!(
            "[bench] appended {} entries → {}",
            results.len(),
            path.display()
        );
    }
    Ok(())
}

#[cfg(feature = "distributed")]
fn run_mpi_bench(
    dag: MpiDagKind,
    scheduler: MpiSchedulerKind,
    n_ops: usize,
    op_duration_ms: u64,
    skew_factor: u64,
    nodes: u32,
    output: Option<PathBuf>,
) -> Result<()> {
    use ferroflow_runtime::{MpiSchedulerMode, MpiWorker};

    let mode = match scheduler {
        MpiSchedulerKind::Static => MpiSchedulerMode::Static,
        MpiSchedulerKind::WorkStealing => MpiSchedulerMode::WorkStealing,
    };
    let sched_name = match mode {
        MpiSchedulerMode::Static => "mpi-static",
        MpiSchedulerMode::WorkStealing => "mpi-work-stealing",
    };
    let (arc_dag, sources, skew) = match dag {
        MpiDagKind::Uniform => {
            let (d, s) = build_mpi_uniform_dag(n_ops, op_duration_ms);
            (d, s, false)
        }
        MpiDagKind::Skewed => {
            let (d, s) = build_mpi_skewed_dag(n_ops, op_duration_ms, skew_factor);
            (d, s, true)
        }
    };
    let dag_size = arc_dag.len();

    match MpiWorker::new(arc_dag, sources).with_mode(mode).run()? {
        Some((_, metrics)) => {
            // Only rank 0 reaches here.
            let run = make_run_metrics(sched_name, skew, nodes, 1, dag_size, metrics);
            println!(
                "[mpi-bench] {sched_name}/{}: elapsed={:.1} ms  throughput={:.0} ops/s  \
                 idle={:.1}%  steal_attempts={}  successful_steals={}  steal_rate={:.1}/s",
                if skew { "skewed" } else { "uniform" },
                run.metrics.elapsed_ms,
                run.metrics.throughput_ops_per_sec,
                idle_pct(&run),
                run.metrics.steal_attempts,
                run.metrics.successful_steals,
                run.metrics.steal_rate,
            );
            if let Some(path) = output {
                append_json(&path, &[run])
                    .with_context(|| format!("writing {}", path.display()))?;
                println!("[mpi-bench] result appended → {}", path.display());
            }
        }
        None => {
            // Worker ranks exit here.
        }
    }
    Ok(())
}

// ── ONNX / synthetic subcommand handlers ──────────────────────────────────────

fn run_info(
    model: Option<&PathBuf>,
    dag: Option<&SyntheticDag>,
    p: &SyntheticDagParams,
) -> Result<()> {
    match (model, dag) {
        (Some(path), None) => {
            let parsed = ferroflow_onnx::parse_onnx(path)
                .with_context(|| format!("parsing {}", path.display()))?;
            println!("{}", dag_summary(&parsed));
        }
        (None, Some(kind)) => {
            let (arc_dag, _) = build_synthetic_dag(kind, p).context("building synthetic DAG")?;
            println!("{}", synthetic_dag_summary(&arc_dag));
        }
        _ => anyhow::bail!("supply exactly one of --model or --dag"),
    }
    Ok(())
}

async fn run_model(
    model: Option<&PathBuf>,
    dag: Option<&SyntheticDag>,
    workers: usize,
    scheduler: OnnxScheduler,
    steal_threshold: usize,
    p: &SyntheticDagParams,
) -> Result<()> {
    let (arc_dag, sources, is_skew, label) = match (model, dag) {
        (Some(path), None) => {
            let (d, s) = load_model(path).with_context(|| format!("loading {}", path.display()))?;
            let arc = std::sync::Arc::new(d);
            println!("{}", dag_summary(&arc));
            (arc, s, false, "onnx".to_string())
        }
        (None, Some(kind)) => {
            let (arc, s) = build_synthetic_dag(kind, p).context("building synthetic DAG")?;
            let skewed = matches!(
                kind,
                SyntheticDag::Wide
                    | SyntheticDag::LargeWide
                    | SyntheticDag::XlargeWide
                    | SyntheticDag::Imbalanced
            );
            let lbl = match kind {
                SyntheticDag::Transformer => "transformer".to_string(),
                SyntheticDag::Wide => format!("wide({}x{},skew={:.2})", p.width, p.depth, p.skew),
                SyntheticDag::Resnet => "resnet".to_string(),
                SyntheticDag::LargeTransformer => {
                    format!("large-transformer({}L,d={})", p.layers, p.d_model)
                }
                SyntheticDag::LargeWide => {
                    format!("large-wide({}x{},skew={:.2})", p.width, p.depth, p.skew)
                }
                SyntheticDag::Imbalanced => format!(
                    "imbalanced({}-heavy×{}ms,{}-fast)",
                    p.n_long_chains,
                    p.chain_depth as u64 * p.slow_factor,
                    p.n_short_ops
                ),
                SyntheticDag::XlargeWide => {
                    format!("xlarge-wide({}x{},skew={:.2})", p.width, p.depth, p.skew)
                }
                SyntheticDag::XlargeTransformer => {
                    format!("xlarge-transformer({}L,d={},h={})", p.layers, p.d_model, p.n_heads)
                }
            };
            println!("{}", synthetic_dag_summary(&arc));
            (arc, s, skewed, lbl)
        }
        _ => anyhow::bail!("supply exactly one of --model or --dag"),
    };

    let dag_size = arc_dag.len();

    let (sched_name, m) = match scheduler {
        OnnxScheduler::Sequential => {
            let (_, m) = SequentialExecutor::execute(&arc_dag, sources)
                .context("sequential executor failed")?;
            ("sequential", m)
        }
        OnnxScheduler::Static => {
            let sched = StaticScheduler::new(&arc_dag, workers);
            let (_, m) = sched
                .execute(std::sync::Arc::clone(&arc_dag), sources)
                .await
                .context("static scheduler failed")?;
            ("static", m)
        }
        OnnxScheduler::WorkStealing => {
            let sched = WorkStealingScheduler::new(workers).with_steal_threshold(steal_threshold);
            let (_, m) = sched
                .execute(std::sync::Arc::clone(&arc_dag), sources)
                .await
                .context("work-stealing scheduler failed")?;
            ("work-stealing", m)
        }
    };

    let run = RunMetrics {
        scheduler: sched_name.to_string(),
        nodes: 1,
        workers_per_node: workers as u32,
        dag_size: dag_size as u32,
        skew: is_skew,
        metrics: m,
    };
    println!(
        "[run] {sched_name}/{label} ({workers}w): {:.1} ms  {:.0} ops/s  idle={:.1}%  steals={:.1}/s",
        run.metrics.elapsed_ms,
        run.metrics.throughput_ops_per_sec,
        idle_pct(&run),
        run.metrics.steal_rate,
    );
    Ok(())
}

// ── entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    match args.command {
        Command::Bench {
            dag,
            workers,
            nodes,
            dag_ops,
            skew_factor,
            chain_dim,
            output,
        } => {
            run_local_bench(dag, workers, nodes, dag_ops, skew_factor, chain_dim, output).await?;
        }

        Command::Report { input } => {
            let raw = std::fs::read_to_string(&input)
                .with_context(|| format!("reading {}", input.display()))?;
            let entries: Vec<RunMetrics> =
                serde_json::from_str(&raw).context("parsing RunMetrics JSON")?;
            print_report(&entries);
        }

        Command::Info {
            model,
            dag,
            seq_len,
            d_model,
            n_heads,
            width,
            depth,
            skew,
            channels,
            layers,
            n_long_chains,
            chain_depth,
            n_short_ops,
            slow_factor,
        } => {
            let p = SyntheticDagParams {
                seq_len,
                d_model,
                n_heads,
                width,
                depth,
                skew,
                channels,
                layers,
                n_long_chains,
                chain_depth,
                n_short_ops,
                slow_factor,
            };
            run_info(model.as_ref(), dag.as_ref(), &p)?;
        }

        Command::Run {
            model,
            dag,
            workers,
            scheduler,
            steal_threshold,
            seq_len,
            d_model,
            n_heads,
            width,
            depth,
            skew,
            channels,
            layers,
            n_long_chains,
            chain_depth,
            n_short_ops,
            slow_factor,
        } => {
            let p = SyntheticDagParams {
                seq_len,
                d_model,
                n_heads,
                width,
                depth,
                skew,
                channels,
                layers,
                n_long_chains,
                chain_depth,
                n_short_ops,
                slow_factor,
            };
            run_model(
                model.as_ref(),
                dag.as_ref(),
                workers,
                scheduler,
                steal_threshold,
                &p,
            )
            .await?;
        }

        #[cfg(feature = "distributed")]
        Command::MpiBench {
            dag,
            scheduler,
            n_ops,
            op_duration_ms,
            skew_factor,
            nodes,
            output,
        } => {
            run_mpi_bench(
                dag,
                scheduler,
                n_ops,
                op_duration_ms,
                skew_factor,
                nodes,
                output,
            )?;
        }
    }

    Ok(())
}
