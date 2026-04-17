use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph, Sparkline},
    Frame,
};

use crate::state::DashboardState;

/// Renders the full four-panel dashboard into `frame`.
pub fn draw(frame: &mut Frame<'_>, state: &DashboardState) {
    let area = frame.size();

    // Split vertically into two rows.
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);

    // Split each row into two columns.
    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(rows[0]);

    let bot = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(rows[1]);

    draw_workers(frame, state, top[0]);
    draw_throughput(frame, state, top[1]);
    draw_steal_activity(frame, state, bot[0]);
    draw_summary(frame, state, bot[1]);
}

fn draw_workers(frame: &mut Frame<'_>, state: &DashboardState, area: Rect) {
    let block = Block::default().title(" Worker Status ").borders(Borders::ALL);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if state.workers.is_empty() {
        return;
    }

    // Give each worker one row; any leftover rows show a summary gauge.
    let n = state.workers.len();
    let gauge_height = 1u16;
    let mut constraints: Vec<Constraint> =
        vec![Constraint::Length(gauge_height); n.min(inner.height as usize)];
    // Add one row for the overall progress bar if room allows.
    if (constraints.len() as u16) < inner.height {
        constraints.push(Constraint::Length(1));
        constraints.push(Constraint::Min(0));
    }
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(inner);

    let max_ops = state.workers.iter().map(|w| w.ops_completed).max().unwrap_or(1).max(1);

    for (i, worker) in state.workers.iter().enumerate() {
        if i >= chunks.len() {
            break;
        }
        let ratio = (worker.ops_completed as f64 / max_ops as f64).clamp(0.0, 1.0);
        let bar_color = match worker.status {
            crate::state::WorkerStatus::Executing => Color::Green,
            crate::state::WorkerStatus::Stealing => Color::Yellow,
            crate::state::WorkerStatus::Idle => Color::DarkGray,
        };
        let label = format!(
            "W{} {} {:4} ops",
            worker.id,
            worker.status.label(),
            worker.ops_completed
        );
        let gauge = Gauge::default()
            .gauge_style(Style::default().fg(bar_color))
            .ratio(ratio)
            .label(label);
        frame.render_widget(gauge, chunks[i]);
    }

    // Overall progress bar.
    let prog_idx = n.min(chunks.len().saturating_sub(2));
    if prog_idx < chunks.len() {
        let gauge = Gauge::default()
            .block(Block::default().title("overall"))
            .gauge_style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
            .ratio(state.progress())
            .label(format!(
                "{}/{} ops",
                state.completed_ops, state.total_ops
            ));
        frame.render_widget(gauge, chunks[prog_idx]);
    }
}

fn draw_throughput(frame: &mut Frame<'_>, state: &DashboardState, area: Rect) {
    let block = Block::default().title(" Throughput (ops/s) ").borders(Borders::ALL);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    // Scale history to u64 for Sparkline (multiply by 10 for sub-integer resolution).
    let data: Vec<u64> = state
        .throughput_history
        .iter()
        .map(|&v| (v * 10.0) as u64)
        .collect();

    let current = state.current_throughput();
    let subtitle = format!("current: {current:.1} ops/s");

    // Split inner: sparkline on top, label on bottom.
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(1)])
        .split(inner);

    let sparkline = Sparkline::default()
        .data(&data)
        .style(Style::default().fg(Color::Green));
    frame.render_widget(sparkline, chunks[0]);

    let label = Paragraph::new(subtitle).style(Style::default().fg(Color::White));
    frame.render_widget(label, chunks[1]);
}

fn draw_steal_activity(frame: &mut Frame<'_>, state: &DashboardState, area: Rect) {
    let rate = if state.elapsed_secs > 0.0 {
        state.successful_steals as f64 / state.elapsed_secs
    } else {
        0.0
    };
    let success_pct = state.steal_success_rate() * 100.0;

    let lines = vec![
        Line::from(vec![
            Span::styled("attempts : ", Style::default().fg(Color::Gray)),
            Span::styled(
                state.steal_attempts.to_string(),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("success  : ", Style::default().fg(Color::Gray)),
            Span::styled(
                state.successful_steals.to_string(),
                Style::default().fg(Color::Green),
            ),
        ]),
        Line::from(vec![
            Span::styled("rate     : ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{rate:.1}/s"),
                Style::default().fg(Color::Yellow),
            ),
        ]),
        Line::from(vec![
            Span::styled("hit%     : ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{success_pct:.1}%"),
                Style::default().fg(Color::Cyan),
            ),
        ]),
    ];

    let para = Paragraph::new(lines)
        .block(Block::default().title(" Steal Activity ").borders(Borders::ALL));
    frame.render_widget(para, area);
}

fn draw_summary(frame: &mut Frame<'_>, state: &DashboardState, area: Rect) {
    let eta = match state.eta_secs() {
        Some(s) if s < 3600.0 => format!("~{s:.1}s"),
        Some(_) => ">1h".into(),
        None => "—".into(),
    };

    let lines = vec![
        Line::from(vec![
            Span::styled("ops      : ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}/{}", state.completed_ops, state.total_ops),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("elapsed  : ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.2}s", state.elapsed_secs),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("idle%    : ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.1}%", state.idle_pct()),
                Style::default().fg(Color::Yellow),
            ),
        ]),
        Line::from(vec![
            Span::styled("ETA      : ", Style::default().fg(Color::Gray)),
            Span::styled(eta, Style::default().fg(Color::Cyan)),
        ]),
    ];

    let para = Paragraph::new(lines)
        .block(Block::default().title(" Summary ").borders(Borders::ALL));
    frame.render_widget(para, area);
}
