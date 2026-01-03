#![allow(dead_code)]
#![allow(unused_imports)]

mod binary_classifier;
mod make_more;

fn main() {
    // binary_classifier::run_all_examples();
    // binary_classifier::binary_classifier();
    make_more::bigram::bigram_example("./files/names.txt".to_string());
}
