use std::collections::VecDeque;

use ferroflow_core::{LiveMetrics, WorkerLiveStatus};

/// Current activity of a single dashboard worker row.
#[derive(Debug, Clone)]
pub enum WorkerStatus {
    Idle,
    Executing,
    Stealing,
}

impl WorkerStatus {
    pub fn label(&self) -> &'static str {
        match self {
            WorkerStatus::Idle => "idle    ",
            WorkerStatus::Executing => "exec    ",
            WorkerStatus::Stealing => "steal   ",
        }
    }
}

/// Per-worker state tracked by the dashboard.
#[derive(Debug, Clone)]
pub struct WorkerState {
    pub id: usize,
    pub status: WorkerStatus,
    pub ops_completed: u64,
    /// Cumulative idle time in milliseconds.
    pub idle_time_ms: f64,
}

/// Full state of the live dashboard, updated on each metrics tick.
pub struct DashboardState {
    pub workers: Vec<WorkerState>,
    /// Throughput samples (ops/s), capped at 60 entries.
    pub throughput_history: VecDeque<f64>,
    pub total_ops: u64,
    pub completed_ops: u64,
    pub elapsed_secs: f64,
    pub steal_attempts: u64,
    pub successful_steals: u64,
    prev_completed: u64,
    prev_elapsed: f64,
}

impl DashboardState {
    /// Creates an initial state for `n_workers` workers.
    pub fn new(n_workers: usize) -> Self {
        Self {
            workers: (0..n_workers)
                .map(|i| WorkerState {
                    id: i,
                    status: WorkerStatus::Idle,
                    ops_completed: 0,
                    idle_time_ms: 0.0,
                })
                .collect(),
            throughput_history: VecDeque::with_capacity(60),
            total_ops: 0,
            completed_ops: 0,
            elapsed_secs: 0.0,
            steal_attempts: 0,
            successful_steals: 0,
            prev_completed: 0,
            prev_elapsed: 0.0,
        }
    }

    /// Ingests a new [`LiveMetrics`] snapshot from the scheduler.
    pub fn update(&mut self, m: &LiveMetrics) {
        self.total_ops = m.total_ops;
        self.completed_ops = m.completed_ops;
        self.elapsed_secs = m.elapsed_secs;
        self.steal_attempts = m.steal_attempts;
        self.successful_steals = m.successful_steals;

        // Compute per-interval throughput.
        let dt = m.elapsed_secs - self.prev_elapsed;
        if dt > 0.0 {
            let delta = m.completed_ops.saturating_sub(self.prev_completed) as f64;
            let sample = delta / dt;
            self.throughput_history.push_back(sample);
            if self.throughput_history.len() > 60 {
                self.throughput_history.pop_front();
            }
        }
        self.prev_completed = m.completed_ops;
        self.prev_elapsed = m.elapsed_secs;

        // Sync per-worker rows.
        for snap in &m.workers {
            if snap.id < self.workers.len() {
                let w = &mut self.workers[snap.id];
                w.ops_completed = snap.ops_completed;
                w.idle_time_ms = snap.idle_us as f64 / 1000.0;
                w.status = match snap.status {
                    WorkerLiveStatus::Executing => WorkerStatus::Executing,
                    WorkerLiveStatus::Stealing => WorkerStatus::Stealing,
                    WorkerLiveStatus::Idle => WorkerStatus::Idle,
                };
            }
        }
    }

    /// Fraction of steal attempts that succeeded; 0 if no attempts yet.
    pub fn steal_success_rate(&self) -> f64 {
        if self.steal_attempts == 0 {
            0.0
        } else {
            self.successful_steals as f64 / self.steal_attempts as f64
        }
    }

    /// Latest throughput sample, or 0 if history is empty.
    pub fn current_throughput(&self) -> f64 {
        self.throughput_history.back().copied().unwrap_or(0.0)
    }

    /// Estimated seconds remaining, or None if no progress yet.
    pub fn eta_secs(&self) -> Option<f64> {
        let throughput = self.current_throughput();
        let remaining = self.total_ops.saturating_sub(self.completed_ops) as f64;
        if throughput > 0.0 {
            Some(remaining / throughput)
        } else {
            None
        }
    }

    /// Fraction of total ops completed, in [0.0, 1.0].
    pub fn progress(&self) -> f64 {
        if self.total_ops == 0 {
            0.0
        } else {
            (self.completed_ops as f64 / self.total_ops as f64).clamp(0.0, 1.0)
        }
    }

    /// Overall idle percentage: idle_ms / (elapsed_ms * n_workers).
    pub fn idle_pct(&self) -> f64 {
        let elapsed_ms = self.elapsed_secs * 1000.0;
        let n = self.workers.len() as f64;
        if elapsed_ms <= 0.0 || n == 0.0 {
            return 0.0;
        }
        let total_idle: f64 = self.workers.iter().map(|w| w.idle_time_ms).sum();
        (total_idle / (elapsed_ms * n) * 100.0).clamp(0.0, 100.0)
    }
}
