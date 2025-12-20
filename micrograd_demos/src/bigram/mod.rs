mod utils;

use std::fs;
use utils::plot_2d_heatmap;

pub fn bigram_example(file_path: String) {
    // Read entire file into a string
    let content = fs::read_to_string(file_path).expect("failed to load the name.txt file");

    // Split by new lines and trim empty lines
    let names = content
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty());

    //
    let mut i = 0;
    let mut ch_lookup: [[u64; 27]; 27] = [[0; 27]; 27];
    for name in names {
        let mut ch1 = '.';
        // Iterate over characters
        for ch2 in name.chars() {
            let ind1 = char_to_index(ch1) % 27;
            let ind2 = char_to_index(ch2) % 27;
            ch_lookup[ind1][ind2] += 1;
            ch1 = ch2;
        }
        i += 1;
        // if i == 100 {
        //     break;
        // }
    }

    let draw_lookup: Vec<Vec<f64>> = ch_lookup.map(|row| row.map(|v| v as f64).to_vec()).to_vec();
    plot_2d_heatmap(&draw_lookup, "heatmap.svg", 1500, 1500, 0).expect("error plotting the image");

    dbg!(ch_lookup);
}

fn char_to_index(c: char) -> usize {
    if c == '.' {
        return 0;
    }
    // b'a' - ASCII value of 'a' as u8
    (c as u8 - b'a' + 1) as usize
}

fn index_to_char(i: usize) -> char {
    if i == 0 {
        return '.';
    }
    (b'a' + i as u8 - 1) as char
}
