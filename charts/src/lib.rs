pub mod heatmap;
pub mod scatter_plot;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plot_2d_heatmap_test() {
        // Sample 10x15 2D data
        let data: Vec<Vec<f64>> = (0..10)
            .map(|i| {
                (0..15)
                    .map(|j| (i as f64 * j as f64).sin() * 10.0 + 20.0) // Arbitrary values
                    .collect()
            })
            .collect();
        let label: Vec<Vec<String>> = (0..10)
            .map(|i| {
                (0..15)
                    .map(|j| format!("{}{}", i, j)) // Arbitrary values
                    .collect()
            })
            .collect();

        let _ = heatmap::plot_2d_heatmap(&data, &label, "./locals/heatmap.svg", 800, 600, 0)
            .map_err(|e| dbg!(e));
        println!("SVG saved as heatmap.svg");
    }

    #[test]
    fn scatter_plot_test() {
        let mut i = -2.5;
        let mut inps = vec![];
        let mut pred = vec![];
        let mut index = 0;
        while i < 3.0 {
            let mut j = -1.0;
            while j < 1.0 {
                inps.push((i, j));
                pred.push((if index % 2 == 0 { 1 } else { -1 }) as f64);
                index += 1;
                j += 0.1;
            }
            i += 0.1;
        }
        let _ = scatter_plot::scatter_plot(
            &inps,
            &pred,
            "Prediction Sample",
            "./locals/scatterplot.svg",
        )
        .map_err(|e| dbg!(e));
    }
}
