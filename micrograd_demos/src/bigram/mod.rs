use std::fs;

pub fn bigram_example(file_path: String) {
    // Read entire file into a string
    let content = fs::read_to_string(file_path).expect("failed to load the name.txt file");

    // Split by new lines and trim empty lines
    let names = content
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty());

    let mut i = 0;
    // Iterate over each name
    for name in names {
        println!("Name: {}", name);

        // Iterate over characters
        for ch in name.chars() {
            println!("  char: {}", ch);
        }
        i += 1;
        if i == 0 {
            break;
        }
    }
}
