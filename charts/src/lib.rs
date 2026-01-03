pub mod heatmap;
pub mod scatter_plot;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heatmap_test() {
        // Sample 10x15 2D data
        let data: Vec<Vec<f64>> = (0..10)
            .map(|i| {
                (0..15)
                    .map(|j| (i as f64 * j as f64).sin() * 10.0 + 20.0) // Arbitrary values
                    .collect()
            })
            .collect();

        heatmap::plot_2d_heatmap(&data, "heatmap.svg", 800, 600, 0);
        println!("SVG saved as heatmap.svg");
    }
}
