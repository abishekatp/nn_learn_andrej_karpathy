use crate::micro_grad::utils::{_scatter_plot, make_moons};
use micrograd::MLP;

pub fn _binary_classifier() {
    let (data, labels) = make_moons(200, 0.1);
    // println!("Data: {:?}\n Labels: {:?}", data,labels);
    if let Err(err) = _scatter_plot(&data, &labels, "Training Sample", "./training_sample.png") {
        dbg!(err);
        return;
    }

    // let _model = MLP::new(2, vec![16, 16, 1]); // 2-layer neural network
}
