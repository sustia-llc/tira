    // Example plotter implementation
    use plotters::prelude::*;
    use plotters::style::Color;
    pub fn plot_recovery(estimates: &[f64], truths: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new("plots/recovery.png", (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Parameter Recovery", ("sans-serif", 50))
        .build_cartesian_2d(0.0..1.0, 0.0..1.0)?;
    
    chart.draw_series(estimates.iter().zip(truths.iter()).map(|(e, t)| {
        Circle::new((*t, *e), 5, BLUE.filled())
    }))?;
    
    Ok(())
}
