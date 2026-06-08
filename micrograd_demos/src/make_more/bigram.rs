use crate::make_more::utils::{char_to_index, index_to_char};

use charts::heatmap::plot_2d_heatmap;
use ndarray::{range, Array2, Axis};
use rand::{distributions::WeightedIndex, prelude::Distribution, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::fs;

pub fn bigram_example(file_path: String, gen_img: bool) {
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
    if gen_img {
        plot_2d_heatmap(
            &draw_lookup,
            &draw_labels,
            "./images/bigram.svg",
            1500,
            1500,
            0,
        )
        .expect("error plotting the image");
    }

    // count_arr - 27x27
    let count_arr = Array2::from_shape_vec(
        (draw_lookup.len(), draw_lookup[0].len()),
        draw_lookup.into_iter().flatten().collect(),
    )
    .expect("expecting to get an array");
    // Adding 1 to all counts for smoothening and avoid zero probabilities.
    let count_arr = count_arr + 1.0;

    // sum_array - 27x1
    let sum_array = count_arr.sum_axis(Axis(1)).insert_axis(Axis(1));

    // prob - (27x27)/(27x1) -> broadcasting into -> (27x27)/(27x27) -> elementwise division
    // (27x1) -> Each element in the first dimension will be repeated for 27 timem.
    let probs = &count_arr / &sum_array;

    /*
    NOTE: the following will print all 1 array.
    let prob_sum = probs.sum_axis(Axis(1));
    dbg!(&prob_sum);
    */

    for _ in 0..10 {
        let name = generate_name(&probs, 100);
        println!("{}", name);
    }
}

fn generate_name(probs: &Array2<f64>, max_len: usize) -> String {
    let seed = 42;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut name = String::new();
    let mut cur_idx = 0;

    loop {
        let prob_vec = probs.row(cur_idx).to_vec();
        let dis = WeightedIndex::new(prob_vec).expect("expecting weighted distribution");
        let next_ind = dis.sample(&mut rng);

        if next_ind == 0 {
            // End token '.'
            break;
        }
        name.push(index_to_char(next_ind));
        cur_idx = next_ind;

        if name.len() == max_len {
            break;
        }
    }
    name
}
