use reproduce::{
    experiment_certainty_weighted, experiment_deterministic, experiment_identical,
    experiment_varying_alpha, experiment_varying_preferences, parameter_recovery_single,
    AifError, RecoveryResult, TrialData,
};
use rayon::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    std::fs::create_dir_all("plots")?;

    let n_trials = 300;

    // -----------------------------------------------------------------------
    // Figure 4: Parameter recovery (§3.1)
    // -----------------------------------------------------------------------
    println!("=== Figure 4: Parameter Recovery ===");

    let true_alphas: Vec<f64> = (1..=40).map(|i| i as f64 * 0.05).collect();
    let recovery_points: Vec<(f64, f64)> = true_alphas
        .par_iter()
        .flat_map(|&true_alpha| {
            (0..5)
                .filter_map(move |_| {
                    parameter_recovery_single(true_alpha, n_trials)
                        .ok()
                        .map(|r| (true_alpha, r.estimated_alpha))
                })
                .collect::<Vec<_>>()
        })
        .collect();

    for &(true_a, est_a) in &recovery_points {
        println!("  true={true_a:.2}  recovered={est_a:.3}");
    }

    // -----------------------------------------------------------------------
    // Figure 5: Four simulation experiments (§3.2)
    // -----------------------------------------------------------------------
    let n_agents_list: [usize; 4] = [4, 8, 16, 100];
    let alpha_steps: Vec<f64> = (1..=20).map(|i| i as f64 * 0.05).collect();

    let exp1_results = run_experiment("Experiment 1: Identical agents", &n_agents_list, &alpha_steps, n_trials, |n, alpha, nt| {
        experiment_identical(n, alpha, nt)
    });

    let exp2_results = run_experiment("Experiment 2: Varying alphas", &n_agents_list, &alpha_steps, n_trials, |n, alpha, nt| {
        experiment_varying_alpha(n, alpha, nt)
    });

    let exp3_results = run_experiment("Experiment 3: Deterministic voting", &n_agents_list, &alpha_steps, n_trials, |n, alpha, nt| {
        experiment_deterministic(n, alpha, nt)
    });

    let exp4_results = run_experiment("Experiment 4: Varying preferences", &n_agents_list, &alpha_steps, n_trials, |n, alpha, nt| {
        experiment_varying_preferences(n, alpha, nt)
    });

    // -----------------------------------------------------------------------
    // Figure 6: Extension — Certainty-weighted voting vs simple voting
    // -----------------------------------------------------------------------
    let exp5_results = run_experiment("Experiment 5: Certainty-weighted voting", &n_agents_list, &alpha_steps, n_trials, |n, alpha, nt| {
        experiment_certainty_weighted(n, alpha, nt)
    });

    // -----------------------------------------------------------------------
    // Generate plots
    // -----------------------------------------------------------------------
    println!("\n=== Generating plots ===");

    plot_figure4(&recovery_points)?;
    plot_figure5(&exp1_results, &exp2_results, &exp3_results, &exp4_results)?;
    plot_figure6(&exp2_results, &exp5_results)?;

    let elapsed = start.elapsed();
    println!("\nDone in {elapsed:.1?}");
    println!("Plots saved to:");
    println!("  plots/figure4_recovery.png");
    println!("  plots/figure5_experiments.png");
    println!("  plots/figure6_certainty_weighted.png");

    Ok(())
}

/// Run an experiment sweep across agent counts and alpha steps.
fn run_experiment<F>(
    name: &str,
    n_agents_list: &[usize; 4],
    alpha_steps: &[f64],
    n_trials: usize,
    experiment_fn: F,
) -> Vec<(f64, f64, usize)>
where
    F: Fn(usize, f64, usize) -> Result<(TrialData, RecoveryResult), AifError>
        + Sync
        + Send,
{
    println!("\n=== {name} ===");
    let f = &experiment_fn;
    let results: Vec<(f64, f64, usize)> = n_agents_list
        .iter()
        .enumerate()
        .flat_map(|(group_idx, &n)| {
            alpha_steps
                .par_iter()
                .filter_map(move |&alpha| {
                    f(n, alpha, n_trials)
                        .ok()
                        .map(|(_, r)| (alpha, r.estimated_alpha, group_idx))
                })
                .collect::<Vec<_>>()
        })
        .collect();

    for &(a, g, gi) in &results {
        println!("  n={:3}  α={a:.2}  group_α={g:.3}", n_agents_list[gi]);
    }
    results
}

// ---------------------------------------------------------------------------
// Plotting
// ---------------------------------------------------------------------------

fn plot_figure4(points: &[(f64, f64)]) -> Result<(), Box<dyn std::error::Error>> {
    use plotters::prelude::*;

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
    use plotters::prelude::*;

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

fn plot_figure5(
    exp1: &[(f64, f64, usize)],
    exp2: &[(f64, f64, usize)],
    exp3: &[(f64, f64, usize)],
    exp4: &[(f64, f64, usize)],
) -> Result<(), Box<dyn std::error::Error>> {
    use plotters::prelude::*;

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
fn plot_figure6(
    exp2: &[(f64, f64, usize)],
    exp5: &[(f64, f64, usize)],
) -> Result<(), Box<dyn std::error::Error>> {
    use plotters::prelude::*;

    let root =
        BitMapBackend::new("plots/figure6_certainty_weighted.png", (1200, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((1, 2));

    plot_panel(&areas[0], "A) Simple voting (Exp 2)", exp2, true)?;
    plot_panel(&areas[1], "B) Certainty-weighted voting", exp5, false)?;

    root.present()?;
    Ok(())
}
