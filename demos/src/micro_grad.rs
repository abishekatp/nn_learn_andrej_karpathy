use micrograd::MVal;

pub fn _run_all_examples() {
    // _micrograd_simple1();
    _micrograd_simple2();
    // _micrograd_clone1();
    // _micrograd_clone2();
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
    o.backward_debug();
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

    o.backward_debug();
}

pub fn _micrograd_clone1() {
    // inputs
    let a = MVal::new_lab(2, "a");
    let mut b = a.clone() + a;
    b.backward_debug();
}

pub fn _micrograd_clone2() {
    // inputs
    let a = MVal::new_lab(-2, "a");
    let b = MVal::new_lab(3, "b");

    let d = a.clone() * b.clone();
    let e = a + b;
    let mut f = d * e;
    f.backward_debug();
}
