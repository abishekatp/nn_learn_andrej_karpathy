use micrograd::Value;

fn main() {
    let a = Value::new(100.0);
    let b = Value::new(30.0);

    println!("a: {}, b: {}", a, b);
    let mut c = a - b;
    Value::backward(&mut c);
    println!("result: {:?}", c);
}
