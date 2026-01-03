use plotters::prelude::*;
use plotters::style::colors::colormaps::ViridisRGB;
pub fn plot_2d_heatmap(
    data: &Vec<Vec<f64>>,  // Your 2D array (rows × cols)
    filename: &str,        // Output SVG file, e.g., "heatmap.svg"
    width: u32,            // Image width in pixels
    height: u32,           // Image height in pixels
    decimal_places: usize, // Number of decimal places for text
) -> Result<(), Box<dyn std::error::Error>> {
    let rows = data.len();
    if rows == 0 {
        return Ok(());
    }
    let cols = data[0].len();

    // Find min and max for normalization
    let mut min_val = data[0][0];
    let mut max_val = data[0][0];
    for row in data {
        for &val in row {
            if val < min_val {
                min_val = val;
            }
            if val > max_val {
                max_val = val;
            }
        }
    }
    let range = max_val - min_val;
    if range == 0.0 {
        return Ok(());
    } // All values identical

    let root = SVGBackend::new(filename, (width, height)).into_drawing_area();
    root.fill(&WHITE)?;

    let cell_w = width as f64 / cols as f64;
    let cell_h = height as f64 / rows as f64;

    // Font settings (adjust size as needed)
    let font_size = (cell_w.min(cell_h) * 0.4) as u32; // ~60% of smaller cell dimension
    let text_style = TextStyle::from(("sans-serif", font_size).into_font());

    for (i, row) in data.iter().enumerate() {
        let y0 = (i as f64 * cell_h) as i32;
        let y1 = ((i + 1) as f64 * cell_h) as i32;

        for (j, &val) in row.iter().enumerate() {
            let x0 = (j as f64 * cell_w) as i32;
            let x1 = ((j + 1) as f64 * cell_w) as i32;

            // Normalize and get color
            let normalized = if range > 0.0 {
                (val - min_val) / range
            } else {
                0.5
            };
            let color: RGBColor = ViridisRGB::get_color(normalized);

            // Draw filled rectangle
            root.draw(&Rectangle::new([(x0, y0), (x1, y1)], color.filled()))?;

            let text = format!("{:.1$}", val, decimal_places);

            // Choose text color: black for light backgrounds, white for dark
            let text_color = if normalized < 0.5 { WHITE } else { BLACK };

            let styled_text = text_style.clone().color(&text_color);

            // Center of the cell
            let center_x = x0 + 2;
            let center_y = (y0 + y1) / 2;

            root.draw_text(&text, &styled_text, (center_x, center_y))?;
        }
    }

    root.present()?;
    Ok(())
}
