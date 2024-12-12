use micrograd::Value;

pub fn _run_all_examples() {
    _micrograd_simple();
    _micrograd_clone1();
    _micrograd_clone2();

    // power example
    // let m = Value::new(2);
    // let w3 = m.clone().pow(5);
    // println!("pow: {}", w3);
}

pub fn _micrograd_simple() {
    // inputs
    let x1 = Value::new(2);
    let x2 = Value::new(0);

    // weights
    let w1 = Value::new(-3);
    let w2 = Value::new(1);

    // bias of the neuron.
    let b = 6.8813735870195432;

    let x1w1 = x1.clone() * w1.clone();
    let x2w2 = x2.clone() * w2.clone();
    let x1w1x2w2 = x1w1 + x2w2;
    let n = x1w1x2w2 + b;
    let mut o = n.tanh();
    o.backward();
    println!("x1:{:?}, w1:{:?}", x1, w1);
    println!("x2:{:?}, w2:{:?}", x2, w2);
}

pub fn _micrograd_clone1() {
    // inputs
    let a = Value::new(2);
    let mut b = a.clone() + a.clone();
    b.backward();
    println!("a:{:?}", a)
}

pub fn _micrograd_clone2() {
    // inputs
    let a = Value::new(-2);
    let b = Value::new(3);

    let d = a.clone() * b.clone();
    let e = a.clone() + b.clone();
    let mut f = d * e;
    f.backward();
    println!("a:{:?}, b:{:?}", a, b)
}
