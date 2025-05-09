use crate::micro_grad::utils::{_scatter_plot, make_moons};
use micrograd::{MVal, MLP};
use plotters::prelude::LogScalable;

pub fn _binary_classifier() {
    let (inps, outs) = make_moons(200, 0.1);
    // println!("Data: {:?}\n Labels: {:?}", data,labels);
    if let Err(err) = _scatter_plot(
        &inps,
        &outs,
        "Training Sample",
        "./images/training_sample.png",
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
            loss = loss + (1.0 - pre.clone() * out.as_f64());

            // accuracy will be high when both output and prediction has the same sign.
            accuracy = accuracy
                + (if pre.get() * (*out).as_f64() > 0.0 {
                    1.0
                } else {
                    0.0
                });
            index += 1;
        }
        let mut avg_loss = loss.clone() / index.as_f64();
        let avg_accr = accuracy * 100.0 / index.as_f64();
        println!("step:{k}, loss:{avg_loss}, accuracy:{avg_accr}%");

        // todo: implement the L2 normalisation
        // _model.parameters()

        // optimisation
        avg_loss.zero_grad();
        avg_loss.backward();

        let learning_rate = 1.0 - (0.9 * k.as_f64() / 100.0);
        for p in model.parameters() {
            p.set(p.get() - MVal::new(learning_rate * p.grad()));
        }
    }

    // make prediction for newly generated data
    let (inps, outs) = make_moons(150, 0.2);
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

    if let Err(err) = _scatter_plot(
        &inps,
        &preds,
        "Prediction Sample",
        "./images/prediction.png",
    ) {
        dbg!(err);
        return;
    }
}
