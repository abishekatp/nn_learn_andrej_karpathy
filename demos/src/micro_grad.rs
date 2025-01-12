use micrograd::{MVal, MLP};

pub fn _run_all_examples() {
    // _micrograd_simple1();
    _micrograd_simple2();
    // _micrograd_clone1();
    // _micrograd_clone2();

    _mlp_example();
}

pub fn _micrograd_simple1() {
    // inputs
    let x1 = MVal::new_lab(2, "x1");
    let x2 = MVal::new_lab(0, "x2");

    // weights
    let w1 = MVal::new_lab(-3, "w1");
    let w2 = MVal::new_lab(1, "w2");

    // bias of the neuron.
    let b = 6.8813735870195432;

    let x1w1 = x1 * w1;
    let x2w2 = x2 * w2;
    let x1w1x2w2 = x1w1 + x2w2;
    let n = x1w1x2w2 + b;
    let mut o = n.tanh();
    // o.backward_debug();
    o.backward();
}

pub fn _micrograd_simple2() {
    // inputs
    let x1 = MVal::new_lab(2, "x1");
    let x2 = MVal::new_lab(0, "x2");

    // weights
    let w1 = MVal::new_lab(-3, "w1");
    let w2 = MVal::new_lab(1, "w2");

    // bias of the neuron.
    let b = 6.8813735870195432;

    let x1w1 = x1 * w1;
    let x2w2 = x2 * w2;
    let x1w1x2w2 = x1w1 + x2w2;
    let n = x1w1x2w2 + b;

    // tanh(x) = (e^2x - 1)/(e^2x + 1)
    let dd = n * 2.0;
    let exp = dd.exp();
    let mut o = (exp.clone() - 1) / (exp + 1);

    // o.backward_debug();
    o.backward();
}

pub fn _micrograd_clone1() {
    // inputs
    let a = MVal::new_lab(2, "a");
    let mut b = a.clone() + a;
    // b.backward_debug();
    b.backward();
}

pub fn _micrograd_clone2() {
    // inputs
    let a = MVal::new_lab(-2, "a");
    let b = MVal::new_lab(3, "b");

    let d = a.clone() * b.clone();
    let e = a + b;
    let mut f = d * e;
    // f.backward_debug();
    f.backward();
}

pub fn _mlp_example() {
    let training_iteration = 2000;
    let mut mlp = MLP::new(3, vec![4, 4, 1]);
    let xs = vec![
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ];
    let ys = vec![1.0, -1.0, -1.0, 1.0];

    println!("\nprediction before training:");
    for x in xs.clone() {
        let ypre = mlp
            .forward(x.to_vec())
            .get(0)
            .unwrap_or(&MVal::new(0))
            .clone();
        println!("ypred:{}", ypre)
    }

    for _ in 0..training_iteration {
        let mut ypred = vec![];
        for x in xs.clone() {
            // last layer has just one neuron.
            let ypre = mlp
                .forward(x.to_vec())
                .get(0)
                .unwrap_or(&MVal::new(0))
                .clone();
            ypred.push(ypre);
        }

        let mut loss = MVal::new(0);
        for (i, ypre) in ypred.into_iter().enumerate() {
            let yact = ys.get(i).unwrap_or(&0.0).clone();
            loss = loss + (ypre.clone() - yact).pow(2);
        }

        loss.zero_grad();
        loss.backward();

        for p in mlp.parameters() {
            let grad = p.grad();
            p.set(p.clone() - 0.05 * grad);
            // println!("w:{}, grad:{}", p, grad)
        }

        // println!("loss: {}\n", loss);
    }

    println!("\nprediction after training:");
    for x in xs.clone() {
        // last layer has just one neuron.
        let ypre = mlp
            .forward(x.to_vec())
            .get(0)
            .unwrap_or(&MVal::new(0))
            .clone();
        println!("ypred:{}", ypre)
    }
}
