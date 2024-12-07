use micrograd::Value;

fn main() {
    micrograd();
}

fn micrograd() {
    // inputs
    let x1 = Value::new(2.0);
    let x2 = Value::new(0.0);

    // weights
    let w1 = Value::new(-3.0);
    let w2 = Value::new(1.0);

    // bias of the neuron.
    let b = Value::new(6.8813735870195432);

    let x1w1 = x1 * w1;
    let x2w2 = x2 * w2;
    let x1w1x2w2 = x1w1 + x2w2;
    let n = x1w1x2w2 + b;
    let mut o = n.tanh();
    o.backward();
}
