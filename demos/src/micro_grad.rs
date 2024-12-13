use micrograd::Value;

pub fn _run_all_examples() {
    // _micrograd_simple1();
    _micrograd_simple2();
    // _micrograd_clone1();
    // _micrograd_clone2();
}

pub fn _micrograd_simple1() {
    // inputs
    let x1 = Value::new_lab(2, "x1");
    let x2 = Value::new_lab(0, "x2");

    // weights
    let w1 = Value::new_lab(-3, "w1");
    let w2 = Value::new_lab(1, "w2");

    // bias of the neuron.
    let b = 6.8813735870195432;

    let x1w1 = x1.clone() * w1.clone();
    let x2w2 = x2.clone() * w2.clone();
    let x1w1x2w2 = x1w1 + x2w2;
    let n = x1w1x2w2 + b;
    let mut o = n.tanh();
    o.backward_debug();
}

pub fn _micrograd_simple2() {
    // inputs
    let x1 = Value::new_lab(2, "x1");
    let x2 = Value::new_lab(0, "x2");

    // weights
    let w1 = Value::new_lab(-3, "w1");
    let w2 = Value::new_lab(1, "w2");

    // bias of the neuron.
    let b = 6.8813735870195432;

    let x1w1 = x1.clone() * w1.clone();
    let x2w2 = x2.clone() * w2.clone();
    let x1w1x2w2 = x1w1 + x2w2;
    let n = x1w1x2w2 + b;

    // tanh(x) = (e^2x - 1)/(e^2x + 1)
    let dd = n.clone() * 2.0;
    let exp = dd.clone().exp();
    let mut o = (exp.clone() - 1) / (exp.clone() + 1);

    o.backward_debug();
}

pub fn _micrograd_clone1() {
    // inputs
    let a = Value::new(2);
    let mut b = a.clone() + a.clone();
    b.backward_debug();
}

pub fn _micrograd_clone2() {
    // inputs
    let a = Value::new(-2);
    let b = Value::new(3);

    let d = a.clone() * b.clone();
    let e = a.clone() + b.clone();
    let mut f = d * e;
    f.backward_debug();
}
