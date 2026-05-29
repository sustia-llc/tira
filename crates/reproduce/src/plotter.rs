#![allow(dead_code)]

use plotters::prelude::*;

/// Data point for parameter recovery / experiment scatter plots.
pub struct ScatterPoint {
    pub x: f64,
    pub y: f64,
    pub group: usize,
}

/// Figure 4: Parameter recovery for α.
/// x = true α, y = inferred α, gray identity line.
pub fn plot_parameter_recovery(
    points: &[ScatterPoint],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (800, 700)).into_drawing_area();
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

    // Identity line
    chart.draw_series(LineSeries::new(
        [(0.0, 0.0), (2.2, 2.2)],
        ShapeStyle::from(RGBColor(180, 180, 180)).stroke_width(1),
    ))?;

    // Scatter points
    chart.draw_series(points.iter().map(|p| {
        Circle::new(
            (p.x, p.y),
            3,
            ShapeStyle::from(RGBColor(50, 100, 180)).filled(),
        )
    }))?;

    root.present()?;
    Ok(())
}

const GROUP_COLORS: [RGBColor; 4] = [
    RGBColor(220, 50, 50),   // red - 4 agents
    RGBColor(50, 100, 200),  // blue - 8 agents
    RGBColor(200, 180, 50),  // yellow - 16 agents
    RGBColor(50, 160, 80),   // green - 100 agents
];

const GROUP_LABELS: [&str; 4] = ["4 agents", "8 agents", "16 agents", "100 agents"];

/// Figure 5: 4-panel experiment results.
/// Each panel: x = mean internal α, y = inferred group α, colored by n_agents.
pub fn plot_experiments(
    panels: &[PanelData; 4],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (1200, 1000)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((2, 2));
    let titles = [
        "A) Simple group",
        "B) Varying alphas",
        "C) Deterministic votes",
        "D) Varying preferences",
    ];

    for (idx, area) in areas.iter().enumerate() {
        let panel = &panels[idx];
        let y_max = panel
            .points
            .iter()
            .map(|p| p.y)
            .fold(1.5_f64, f64::max)
            .min(5.0);

        let mut chart = ChartBuilder::on(area)
            .caption(titles[idx], ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(35)
            .y_label_area_size(45)
            .build_cartesian_2d(0.0..1.1, 0.0..y_max)?;

        chart
            .configure_mesh()
            .x_desc("(Mean of) internal α")
            .y_desc("Inferred group α")
            .draw()?;

        // Identity line
        chart.draw_series(LineSeries::new(
            [(0.0, 0.0), (1.1, 1.1)],
            ShapeStyle::from(RGBColor(180, 180, 180)).stroke_width(1),
        ))?;

        // Draw each group (n_agents) in a different color
        for group_idx in 0..4 {
            let group_points: Vec<&ScatterPoint> = panel
                .points
                .iter()
                .filter(|p| p.group == group_idx)
                .collect();

            if group_points.is_empty() {
                continue;
            }

            let color = GROUP_COLORS[group_idx];
            chart
                .draw_series(group_points.iter().map(|p| {
                    Circle::new((p.x, p.y), 4, ShapeStyle::from(color).filled())
                }))?
                .label(GROUP_LABELS[group_idx])
                .legend(move |(x, y)| Circle::new((x, y), 4, ShapeStyle::from(color).filled()));
        }

        if idx == 1 {
            chart
                .configure_series_labels()
                .position(SeriesLabelPosition::UpperRight)
                .background_style(WHITE.mix(0.8))
                .border_style(BLACK)
                .draw()?;
        }
    }

    root.present()?;
    Ok(())
}

pub struct PanelData {
    pub points: Vec<ScatterPoint>,
}

impl PanelData {
    #[must_use]
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    pub fn push(&mut self, x: f64, y: f64, group: usize) {
        self.points.push(ScatterPoint { x, y, group });
    }
}

impl Default for PanelData {
    fn default() -> Self {
        Self::new()
    }
}
