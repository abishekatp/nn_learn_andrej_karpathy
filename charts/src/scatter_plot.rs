use ndarray::Array2;
use plotters::prelude::*;

pub fn scatter_plot(
    data: &Array2<f64>,
    category: &Vec<f64>,
    title: &str,
    file_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // "./scatter_plot.png"
    let root_area = BitMapBackend::new(file_path, (1024, 768)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let root_area = root_area.titled(title, ("sans-serif", 60))?;

    let mut cc = ChartBuilder::on(&root_area)
        .margin(5)
        .set_all_label_area_size(50)
        .build_cartesian_2d(-2.5f32..3.0, -1.2f32..1.2f32)?;
    cc.configure_mesh()
        .x_labels(20)
        .y_labels(10)
        .disable_mesh()
        .x_label_formatter(&|v| format!("{:.1}", v))
        .y_label_formatter(&|v| format!("{:.1}", v))
        .draw()?;

    // Draw the scatter points
    let mut i = 0;
    cc.draw_series(data.rows().into_iter().map(|row| {
        let x = row
            .get(0)
            .expect("expecting the x axis value in the array 2nd dimention")
            .clone() as f32;
        let y = row
            .get(1)
            .expect("expecting the x axis value in the array 2nd dimention")
            .clone() as f32;
        let cat = category
            .get(i)
            .expect("expectinve category number each data point")
            .clone();
        i += 1;
        Circle::new(
            (x, y),
            5,
            ShapeStyle {
                // here setting the negative and positive limits to as much close to -1 and 1
                // as possible will show the most accurate predictions.
                color: if cat < -0.5 {
                    BLUE.to_rgba()
                } else if cat > 0.5 {
                    GREEN.to_rgba()
                } else {
                    RED.to_rgba()
                },
                filled: true,
                stroke_width: 1,
            },
        )
    }))?;

    // Add labels to points (optional)
    // for r in data.rows() {
    //     let x = r.get(0).unwrap().clone() as f32;
    //     let y = r.get(1).unwrap().clone() as f32;
    //     let label = format!("({:.1}, {:.1})", x, y);
    //     cc.draw_series([Text::new(label, (x, y), ("sans-serif", 12).into_font())])?;
    // }

    Ok(())
}
