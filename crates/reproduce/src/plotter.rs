//! Figure rendering for the `reproduce` binary.
//!
//! Consolidated per #7: these were previously duplicated — a stale, uncalled copy
//! lived here while `bin/reproduce.rs` carried the live, diverged version it actually
//! called. This module now holds the one live copy; the binary is orchestration only
//! (compute the data, then call these render functions).
//!
//! - [`plot_figure4`] — Figure 4 (parameter recovery scatter, single panel).
//! - [`plot_figure5`] — Figure 5 (four-panel experiment grid).
//! - [`plot_figure6`] — Figure 6 (certainty-weighted vs simple voting, two panels).
//! - `plot_panel` — shared per-panel scatter/legend helper behind Figures 5 and 6
//!   (private: it takes a raw `DrawingArea` sub-region, not a top-level path, so it
//!   has no standalone caller outside this module).

use plotters::prelude::*;

/// Figure 4: Parameter recovery for α.
/// x = true α, y = inferred α, gray identity line.
pub fn plot_figure4(points: &[(f64, f64)]) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("plots/figure4_recovery.png", (800, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Parameter recovery for α", ("sans-serif", 28))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0..2.2, 0.0..5.0)?;

    chart
        .configure_mesh()
        .x_desc("True α")
        .y_desc("Inferred α")
        .draw()?;

    chart.draw_series(LineSeries::new(
        [(0.0, 0.0), (2.2, 2.2)],
        ShapeStyle::from(RGBColor(180, 180, 180)).stroke_width(1),
    ))?;

    chart.draw_series(points.iter().map(|&(x, y)| {
        Circle::new((x, y), 3, ShapeStyle::from(RGBColor(50, 100, 180)).filled())
    }))?;

    root.present()?;
    Ok(())
}

const COLORS: [plotters::style::RGBColor; 4] = [
    plotters::style::RGBColor(220, 50, 50),
    plotters::style::RGBColor(50, 100, 200),
    plotters::style::RGBColor(200, 180, 50),
    plotters::style::RGBColor(50, 160, 80),
];
const LABELS: [&str; 4] = ["4 agents", "8 agents", "16 agents", "100 agents"];

fn plot_panel(
    area: &plotters::prelude::DrawingArea<plotters::prelude::BitMapBackend<'_>, plotters::coord::Shift>,
    title: &str,
    data: &[(f64, f64, usize)],
    show_legend: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let y_max = data
        .iter()
        .map(|&(_, y, _)| y)
        .fold(1.5_f64, f64::max)
        .min(5.0);

    let mut chart = ChartBuilder::on(area)
        .caption(title, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(35)
        .y_label_area_size(45)
        .build_cartesian_2d(0.0..1.1, 0.0..y_max)?;

    chart
        .configure_mesh()
        .x_desc("(Mean of) internal α")
        .y_desc("Inferred group α")
        .draw()?;

    chart.draw_series(LineSeries::new(
        [(0.0, 0.0), (1.1, 1.1)],
        ShapeStyle::from(RGBColor(180, 180, 180)).stroke_width(1),
    ))?;

    for gi in 0..4 {
        let group_pts: Vec<(f64, f64)> = data
            .iter()
            .filter(|&&(_, _, g)| g == gi)
            .map(|&(x, y, _)| (x, y))
            .collect();

        if group_pts.is_empty() {
            continue;
        }

        let color = COLORS[gi];
        chart
            .draw_series(
                group_pts
                    .iter()
                    .map(|&(x, y)| Circle::new((x, y), 4, ShapeStyle::from(color).filled())),
            )?
            .label(LABELS[gi])
            .legend(move |(x, y)| Circle::new((x, y), 4, ShapeStyle::from(color).filled()));
    }

    if show_legend {
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperLeft)
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;
    }

    Ok(())
}

/// Figure 5: 4-panel experiment results.
/// Each panel: x = mean internal α, y = inferred group α, colored by n_agents.
pub fn plot_figure5(
    exp1: &[(f64, f64, usize)],
    exp2: &[(f64, f64, usize)],
    exp3: &[(f64, f64, usize)],
    exp4: &[(f64, f64, usize)],
) -> Result<(), Box<dyn std::error::Error>> {
    let root =
        BitMapBackend::new("plots/figure5_experiments.png", (1200, 1000)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((2, 2));
    let titles = [
        "A) Simple group",
        "B) Varying alphas",
        "C) Deterministic votes",
        "D) Varying preferences",
    ];
    let datasets: [&[(f64, f64, usize)]; 4] = [exp1, exp2, exp3, exp4];

    for (idx, area) in areas.iter().enumerate() {
        plot_panel(area, titles[idx], datasets[idx], idx == 0)?;
    }

    root.present()?;
    Ok(())
}

/// Figure 6: Simple probabilistic voting (Exp 2) vs certainty-weighted voting (Exp 5).
/// Side-by-side comparison — same Dirichlet-constructed varying α, different aggregation.
pub fn plot_figure6(
    exp2: &[(f64, f64, usize)],
    exp5: &[(f64, f64, usize)],
) -> Result<(), Box<dyn std::error::Error>> {
    let root =
        BitMapBackend::new("plots/figure6_certainty_weighted.png", (1200, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((1, 2));

    plot_panel(&areas[0], "A) Simple voting (Exp 2)", exp2, true)?;
    plot_panel(&areas[1], "B) Certainty-weighted voting", exp5, false)?;

    root.present()?;
    Ok(())
}
