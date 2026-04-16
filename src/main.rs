use anyhow::Result;
use clap::{Parser, ValueEnum};

#[derive(Debug, Clone, ValueEnum)]
enum Scheduler {
    Sequential,
    WorkStealing,
}

/// ferroflow — distributed work-stealing tensor computation scheduler
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Scheduler strategy
    #[arg(short, long, default_value = "sequential")]
    scheduler: Scheduler,

    /// Path to DAG spec file (JSON)
    #[arg(short, long)]
    dag_file: Option<std::path::PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let _args = Args::parse();
    tracing::info!("ferroflow starting");
    Ok(())
}
