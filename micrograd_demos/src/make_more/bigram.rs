use charts::heatmap::plot_2d_heatmap;

use crate::make_more::utils::{char_to_index, index_to_char};
use std::fs;

pub fn bigram_example(file_path: String) {
    // Read entire file into a string
    let content = fs::read_to_string(file_path).expect("failed to load the name.txt file");

    // Split by new lines and trim empty lines
    let names = content
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty());

    // create char lookup table
    let mut ch_lookup: [[u64; 27]; 27] = [[0; 27]; 27];
    for name in names {
        // dbg!(&name);
        let mut ch1 = '.';
        // Iterate over characters
        for ch2 in name.chars() {
            let ind1 = char_to_index(ch1) % 27;
            let ind2 = char_to_index(ch2) % 27;
            ch_lookup[ind1][ind2] += 1;
            ch1 = ch2;
        }
    }

    // draw the lookup table as heatmap
    let draw_lookup: Vec<Vec<f64>> = ch_lookup.map(|row| row.map(|v| v as f64).to_vec()).to_vec();
    let draw_labels: Vec<Vec<String>> = ch_lookup
        .iter()
        .enumerate()
        .map(|(i, row)| {
            row.iter()
                .enumerate()
                .map(|(j, _)| format!("{}{}", index_to_char(i), index_to_char(j)))
                .collect()
        })
        .collect();
    plot_2d_heatmap(
        &draw_lookup,
        &draw_labels,
        "./images/heatmap.svg",
        1500,
        1500,
        0,
    )
    .expect("error plotting the image");
}
