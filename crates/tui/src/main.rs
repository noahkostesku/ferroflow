mod state;
mod ui;

use std::io;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use clap::Parser;
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ferroflow_core::{Dag, LiveMetrics, RunMetrics};
use ferroflow_runtime::WorkStealingScheduler;
use ratatui::{backend::CrosstermBackend, Terminal};
use tokio::sync::watch;

use state::DashboardState;

#[derive(Parser)]
#[command(name = "ferroflow-tui", about = "Live dashboard for work-stealing execution")]
struct Args {
    /// Number of worker tasks.
    #[arg(long, short, default_value_t = 4)]
    workers: usize,

    /// Number of ops per branch in the skewed DAG (must be even, ≥ 2).
    #[arg(long, default_value_t = 40)]
    dag_size: usize,

    /// Slow-branch sleep in ms per op (100 → ~2.5 s wall time with 4 workers).
    #[arg(long, default_value_t = 100)]
    skew: u64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Build skewed DAG.
    let (dag, sources) = Dag::with_skew(args.dag_size, args.skew)
        .context("failed to build DAG")?;
    let dag = Arc::new(dag);
    let n_workers = args.workers;
    let dag_size = dag.ops.len() as u32;
    let total_ops = dag.ops.iter().filter(|op| !op.input_ids.is_empty()).count() as u64;

    // Setup terminal.
    enable_raw_mode().context("enable raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).context("enter alternate screen")?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("create terminal")?;

    // Watch channel: scheduler → TUI.
    let init = LiveMetrics::empty(n_workers);
    let (tx, mut rx) = watch::channel(init);

    // Spawn scheduler task.
    let sched = WorkStealingScheduler::new(n_workers);
    let dag_clone = Arc::clone(&dag);
    let sched_handle = tokio::spawn(async move {
        sched.execute_with_watch(dag_clone, sources, tx).await
    });

    // Dashboard state.
    let mut dash = DashboardState::new(n_workers);
    dash.total_ops = total_ops;

    let mut quit = false;

    // TUI event loop — runs until scheduler finishes or user presses q/Esc.
    loop {
        // Non-blocking key event check.
        if event::poll(Duration::ZERO).unwrap_or(false) {
            if let Ok(Event::Key(key)) = event::read() {
                if matches!(key.code, KeyCode::Char('q') | KeyCode::Esc) {
                    quit = true;
                    break;
                }
            }
        }

        // Drain latest metrics snapshot.
        if rx.has_changed().unwrap_or(false) {
            let live = rx.borrow_and_update().clone();
            dash.update(&live);
        }

        terminal.draw(|f| ui::draw(f, &dash))?;

        if sched_handle.is_finished() {
            // One final update so the dashboard shows 100%.
            if rx.has_changed().unwrap_or(false) {
                let live = rx.borrow_and_update().clone();
                dash.update(&live);
            }
            terminal.draw(|f| ui::draw(f, &dash))?;
            // Hold the final frame so the user can see the completed state.
            tokio::time::sleep(Duration::from_millis(1500)).await;
            break;
        }

        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Teardown terminal.
    disable_raw_mode().ok();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).ok();
    terminal.show_cursor().ok();

    if quit {
        eprintln!("Interrupted.");
        return Ok(());
    }

    // Collect results and print final RunMetrics table.
    match sched_handle.await? {
        Ok((_results, metrics)) => {
            let run = RunMetrics {
                scheduler: "work-stealing".into(),
                nodes: 1,
                workers_per_node: n_workers as u32,
                dag_size,
                skew: true,
                metrics,
            };
            println!("\n── Final Run Metrics ──────────────────────────────");
            println!("{run}");
            println!("───────────────────────────────────────────────────");
        }
        Err(e) => eprintln!("Scheduler error: {e}"),
    }

    Ok(())
}
