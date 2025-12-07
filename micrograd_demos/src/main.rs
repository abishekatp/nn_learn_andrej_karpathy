#![allow(dead_code)]
#![allow(unused_imports)]

mod bigram;
mod micro_grad;

fn main() {
    // micro_grad::run_all_examples();
    // micro_grad::binary_classifier();
    bigram::bigram_example("./files/names.txt".to_string());
}
