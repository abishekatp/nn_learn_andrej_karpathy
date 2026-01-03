use crate::binary_classifier::utils::make_moons;
use charts::scatter_plot::scatter_plot;
use micrograd::{MVal, MLP};
use ndarray::Array2;

pub fn binary_classifier() {
    let seed = 42;
    let (inps, outs) = make_moons(200, 0.1, seed);
    // println!("Data: {:?}\n Labels: {:?}", data,labels);
    let draw_inp = inps
        .rows()
        .into_iter()
        .map(|row| {
            (
                row.get(0).unwrap_or(&0.0).clone(),
                row.get(1).unwrap_or(&0.0).clone(),
            )
        })
        .collect::<Vec<(f64, f64)>>();
    if let Err(err) = scatter_plot(
        &draw_inp,
        &outs,
        "Training Sample",
        "./images/training_sample.svg",
    ) {
        dbg!(err);
        return;
    }

    // first layer will have 16 neurons each with two inputs. first layer will have 16 ouputs.
    // second layer will have 16 neurons wach with 16 inputs. second layer will have 16 outputs.
    // third layer will have 1 neuron with 16 inputs. third layer will have single output.
    let mut model = MLP::new(2, vec![16, 16, 1]); // 2-layer neural network

    println!("\ntraining the model:");
    for k in 0..100 {
        let mut index = 0;
        let mut loss = MVal::new(0.0);
        let mut accuracy = 0.0;
        for row in inps.rows() {
            let v = row.to_vec();
            // forward the model to get the prediction
            // ypre is expected to be approximately equal to -1 or 1. This is what we are training the model for.
            let pre = model
                .forward(v)
                .get(0)
                .expect("expecting single output since the last layer has single neuron")
                .clone();
            // println!("pred:{pre}");

            // out will be either -1 or 1
            let out = outs
                .get(index)
                .expect("expecting the output label for each input");
            // svm "max-margin" loss. loss will increase when ouput and prediction is not matching
            loss = loss + (1.0 - pre.clone() * *out);

            // accuracy will be high when both output and prediction has the same sign.
            accuracy = accuracy + (if pre.get() * (*out) > 0.0 { 1.0 } else { 0.0 });
            index += 1;
        }
        let mut avg_loss = loss.clone() / (index as f64);
        let avg_accr = accuracy * 100.0 / (index as f64);
        println!("step:{k}, loss:{avg_loss}, accuracy:{avg_accr}%");

        // todo: implement the L2 normalisation
        // _model.parameters()

        // optimisation
        avg_loss.zero_grad();
        avg_loss.backward();

        let learning_rate = 1.0 - (0.9 * (k as f64) / 100.0);
        for p in model.parameters() {
            p.set(p.get() - MVal::new(learning_rate * p.grad()));
        }
    }

    test_for_all_points(&model);
    test_for_generated(&model);
}

fn test_for_generated(model: &MLP) {
    // make prediction for newly generated data
    let seed = 142;
    let (inps, outs) = make_moons(150, 0.5, seed);
    println!("\nmaking the predictions:");
    let mut index = 0;
    let mut preds = vec![];
    for row in inps.rows() {
        let v = row.to_vec();
        // forward the model to get the prediction
        // ypre is expected to be approximately equal to -1 or 1. This is what we are training the model for.
        let pre = model
            .forward(v)
            .get(0)
            .expect("expecting single output since the last layer has single neuron")
            .clone();
        preds.push(pre.get());

        let out = outs
            .get(index)
            .expect("expecting the output label for each input");

        println!("pred:{pre}, out:{out}");
        index += 1;
    }

    let draw_inp = inps
        .rows()
        .into_iter()
        .map(|row| {
            (
                row.get(0).unwrap_or(&0.0).clone(),
                row.get(1).unwrap_or(&0.0).clone(),
            )
        })
        .collect::<Vec<(f64, f64)>>();

    if let Err(err) = scatter_plot(
        &draw_inp,
        &preds,
        "Prediction Sample",
        "./images/prediction1.svg",
    ) {
        dbg!(err);
        return;
    }
}

fn test_for_all_points(model: &MLP) {
    // make prediction for newly generated data
    let inps = get_all_test_points();
    println!("\nmaking the predictions:");

    let mut preds = vec![];
    for v in inps.clone() {
        // forward the model to get the prediction
        // ypre is expected to be approximately equal to -1 or 1. This is what we are training the model for.
        let pre = model
            .forward(v)
            .get(0)
            .expect("expecting single output since the last layer has single neuron")
            .clone();
        preds.push(pre.get());

        println!("pred:{pre}");
    }

    let draw_inp = inps.iter().map(|v| (v[0], v[1])).collect();
    if let Err(err) = scatter_plot(
        &draw_inp,
        &preds,
        "Prediction Sample",
        "./images/prediction2.svg",
    ) {
        dbg!(err);
        return;
    }
}

fn get_all_test_points() -> Vec<Vec<f64>> {
    let mut i = -2.5;
    let mut inps = vec![];
    while i < 3.0 {
        let mut j = -1.0;
        while j < 1.0 {
            inps.push(vec![i, j]);
            j += 0.1;
        }
        i += 0.1;
    }
    inps
}
