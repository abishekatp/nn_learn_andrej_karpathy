use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::Debug,
    hash::{Hash, Hasher},
    rc::Rc,
};

use data_type::{DataType, IntoValue};

pub mod add;
pub mod data_type;
pub mod display;
pub mod div;
pub mod mul;
pub mod others;
pub mod sub;

/*
What is Value object:
  Value is a fundamental unit of the neural network. All the logic is build
on top of this Value Object. It will have the following properties.
    - The ultimate purpose of a Value is storing some floating point number in the
      `data` field and computing it gradient value w.r.t to some arbitrary output Value that
      being computed using the Value.

    - Each operation(like addition, multiplication, division of two Value objects) that is
      implemented on the Value will have its own corresponding gradient implementation. This
      gradient implementation will compute the gradient of the Value object that is being
      used with the particular operation(e.g addition, multiplication, etc) to compute some
      output Value.

    - The gradient will tell you, how the small change in the `data` field of the Value
      w.r.t a particular operation will affect the output Value's `data` field.
      But remember that the `grad` field stores the global gradient of the Value.

    - The global gradient of the Value will be equal to the
      gradinet of the current Value multiplied by the gradient of the output Value.
      This multiplication will continue until we reach the final output Value. Sicne
      the output Value of the current operation might be used in another computation
      together with some other operation. This is based on the chain rule of derivation.

    - The global gradient will tell us how small change to the `data` field of the current Value
      will affect the final output Value's `data` field. The final output will be depending on
      the current Value either directly or indirectly.

Use case:
    - We can construct a complex expression using multiple such Values and it's implemented operations,
      then we can copute the global gradient of each of these Values using the back propogation.

    - The Value is wrapped with Rc and RefCell to be able clone and reuse the same
      Value in a computation of different output Values. The `grad` field will store the
      sum of the gradient values computed for each instance of the usage of the Value.

    - Suppose we have the Value A, B, C and E. C = A + B, D = A * C and E = C + D. then the gradient
      of A will be sum of gradient w.r.t C and D. This is because A will affect the final
      output Value E in two ways. The first way is through the addition with B and the second way
      is through the multiplication with C.



The chain rule of derivation can be explained as follows:
    equation-1 -> C = A + B
    equation-2 -> F = (D * C) + E
    equation-3 -> O = F * G

- Then the local derivative of equation-1 is dC/dA = 1 and dC/dB = 1. Similarly for equation-2 are
dF/dD = C, dF/dC = D and dF/dE = 1. Similarly for equation-3 dO/dF = G and dO/dG = F.
- Then global gradient of A with respect to the final output O is defined as follow,

dO/dA => dO/dF * dF/dC * dC/dA = G * D * 1

- Similarly we can compute the global gradient of any variable in the the above equations.

- The gradient of addition operator can be derived as follows. Suppose in the equation-1 there
is a small change h in the input value A, then how will that affect the output C.

C1 = A + B
C2 = (A + h) + B
dC/dA = (C2 - C1)/((A+h)-A) = (((A+h)+B) - (A+B)) / h = (A+h+B-A-B)/h = 1

- Similarly we can derive the gradient of multiplication and other operators also.
*/

#[derive(Debug)]
enum Operator {
    None,
    Plus,
    Minus,
    Mul,
    Div,
    Tanh,
    ReLU,
    Exp,
    Pow,
}

pub struct Value {
    data: DataType,
    grad: DataType, // grad(global gradient) field will have gradient of final output with respect to the Value(Self).
    operands: Vec<MVal>,
    operator: Operator,
    // for debugging purpose
    label: String,
}

/// If you apply any of the +, -, *, / operation on the MVal, then it will return the new instance of MVal with the result.
/// Remember that assignment operator will make the Mval instance to point to the new address.
/// we need Rc, because we need to allow reuse of instance of MVal in multiple places.
/// we need RefCell, because we need to mutate the grad field in the Value for each MVal
/// Cloning of MVal(MutableValue) is a cheap operation since it is acutally wrapper around Rc and RefCell
/// because of the same if you try to use the same Value in two places, then
/// it's gradient will be the accumulated sum of gradients of all the places it has been used.
#[derive(Clone)]
pub struct MVal(Rc<RefCell<Value>>);

impl MVal {
    pub fn new<T: IntoValue>(data: T) -> Self {
        Self(Rc::new(RefCell::new(Value::new(data))))
    }
    pub fn new_lab<T: IntoValue>(data: T, label: &str) -> Self {
        Self(Rc::new(RefCell::new(Value::new_lab(data, label))))
    }
    pub fn default() -> Self {
        Self(Rc::new(RefCell::new(Value::new(0))))
    }

    pub fn grad(&self) -> DataType {
        self.0.borrow().grad
    }
    /// to mutate the gradient inplace instead creating new instance.
    pub fn set_grad(&self, val: MVal) {
        self.0.borrow_mut().grad = val.grad();
    }

    pub fn get(&self) -> DataType {
        self.0.borrow().data
    }
    /// to mutate the data inplace instead creating new instance.
    pub fn set(&self, val: MVal) {
        self.0.borrow_mut().data = val.get();
    }
}

impl Value {
    /// Cloning of returned value(MVal) is a cheap operation since it is acutally wrapper around Rc and RefCell
    pub fn new<T: IntoValue>(data: T) -> Self {
        Self {
            data: data.into_value(),
            grad: 0.0,
            operands: vec![],
            operator: Operator::None,
            label: String::new(),
        }
    }
    /// new with label is for the debugging process that assigns label for each Value.
    pub fn new_lab<T: IntoValue>(data: T, label: &str) -> Self {
        Self {
            data: data.into_value(),
            grad: 0.0,
            operands: vec![],
            operator: Operator::None,
            label: label.to_string(),
        }
    }

    /// calling compute gradient on a output value will comput the gradient value
    /// w.r.t each one of its operands and store those gradients in the corresponding operands objects.
    /// Here chain rule of derivative is used: dy/dz = (dy/dx)(dx/dz)
    /// updated the gradient values inplace.
    fn comput_gradient(&mut self) {
        /*
             If you use the same Value in two places, then it's gradient will be
             the accumulated sum of gradients of all the places it has been used.
        */
        let out = self;
        match out.operator {
            Operator::Plus => {
                if out.operands.len() == 2 {
                    let mut lhs = out.operands[0].0.borrow_mut();
                    /*
                        if you use borrow_mut just next to each other, then it will panic
                        when we use let c = a.clone() + a; because we trying to get the
                        mutable reference of same variable two times
                    */
                    if let Ok(mut rhs) = out.operands[1].0.try_borrow_mut() {
                        // if y = x + z, then dy/dx = 1 and dy/dz = 1
                        lhs.grad += out.grad * 1.0;
                        rhs.grad += out.grad * 1.0;
                    } else {
                        /*
                            since this is single threaded only other place we got the
                            mutable reference is just in the above line.

                            if y = x + x = 2x then dy/dx = 2;
                        */
                        lhs.grad += out.grad * 2.0;
                    }
                }
            }
            Operator::Minus => {
                if out.operands.len() == 2 {
                    let mut lhs = out.operands[0].0.borrow_mut();
                    // handle: let c = a.clone() - a;
                    if let Ok(mut rhs) = out.operands[1].0.try_borrow_mut() {
                        // if y = x - z, then dy/dx = 1 and dy/dz = -1
                        lhs.grad += out.grad * 1.0;
                        rhs.grad += out.grad * -1.0;
                    }
                    // else case: if y = x - x, then dy/dx = 0
                }
            }
            Operator::Mul => {
                if out.operands.len() == 2 {
                    let mut lhs = out.operands[0].0.borrow_mut();
                    if let Ok(mut rhs) = out.operands[1].0.try_borrow_mut() {
                        lhs.grad += out.grad * rhs.data;
                        rhs.grad += out.grad * lhs.data;
                    } else {
                        // if y = x * x = x^2, then dy/dx = 2x
                        lhs.grad += out.grad * (2.0 * lhs.data);
                    }
                }
            }
            Operator::Div => {
                if out.operands.len() == 2 {
                    let mut numerator = out.operands[0].0.borrow_mut();

                    // y=x/z then dy/dx = 1/z and dy/dz = -x/y^2.
                    if let Ok(mut denominator) = out.operands[1].0.try_borrow_mut() {
                        numerator.grad += out.grad * 1.0 / denominator.data;
                        denominator.grad += (out.grad * -1.0 * numerator.data)
                            / (denominator.data * denominator.data);
                    }
                    // else case: if y = x/x, then dy/dx = 0
                }
            }
            Operator::Tanh => {
                if out.operands.len() == 1 {
                    let mut input = out.operands[0].0.borrow_mut();

                    // y = tahh(x) then dy/dx = 1 - (tanh(x))^2
                    input.grad += out.grad * (1.0 - (out.data * out.data));
                }
            }
            Operator::ReLU => {
                if out.operands.len() == 1 {
                    let mut input = out.operands[0].0.borrow_mut();

                    // ReLU y = max(0,x). dy/dx = 1 for x > 0.
                    input.grad += out.grad * (if out.data > 0.0 { 1.0 } else { 0.0 });
                }
            }
            Operator::Exp => {
                if out.operands.len() == 1 {
                    let mut input = out.operands[0].0.borrow_mut();

                    // y = exp(x) then dy/dx = exp(x)
                    input.grad += out.grad * out.data;
                }
            }
            Operator::Pow => {
                if out.operands.len() == 2 {
                    let mut input = out.operands[0].0.borrow_mut();
                    let powv = out.operands[1].0.borrow().data;

                    // y = x^n then dy/dx = n * x^(n-1)
                    input.grad += out.grad * (powv * input.data.powf(powv - 1.0));
                }
            }
            Operator::None => {}
        }
    }
}

impl MVal {
    /// update the gradient values inplace.
    pub fn backward_debug(&mut self) {
        let mut all_nodes = self.collect_operands();
        all_nodes.reverse();

        // set the gradient of the root node to 1 since gradient with itself is 1.
        if let Some(first) = all_nodes.get(0) {
            let mut val = first.0.borrow_mut();
            val.grad = 1.0;

            val.comput_gradient();
            println!("{:?}", val)
        }
        for i in 1..all_nodes.len() {
            if let Some(el) = all_nodes.get(i) {
                let mut val = el.0.borrow_mut();
                val.comput_gradient();
                println!("{:?}", val)
            }
        }
    }

    /// update the gradient values inplace.
    pub fn backward(&mut self) {
        let mut all_nodes = self.collect_operands();
        all_nodes.reverse();

        // set the gradient of the root node to 1 since gradient with itself is 1.
        if let Some(first) = all_nodes.get(0) {
            let mut val = first.0.borrow_mut();
            val.grad = 1.0;

            val.comput_gradient();
        }
        for i in 1..all_nodes.len() {
            if let Some(el) = all_nodes.get(i) {
                let mut val = el.0.borrow_mut();
                val.comput_gradient();
            }
        }
    }

    pub fn collect_operands(&self) -> Vec<MVal> {
        let mut visited = HashSet::new();
        self.collect_operands_inner(&mut visited)
    }

    /// collect all the Values that a calling Values depending on
    /// in the breadth first search(BFS) manner.
    /*
      We have to use the topological order here. otherwise if the same Value is
      re-used two times in two different places, then all its nested operands(children) will be
      called twice. To avoide this issue we use topological order and check weather each node is
      visited before exploring all its children.
    */
    // todo: here I could generate some unique id field for each MVal and use it keep track of visisted set.
    pub fn collect_operands_inner(&self, visited: &mut HashSet<MVal>) -> Vec<MVal> {
        let mut all_nodes = vec![];

        // if not already visited, then collect all of its children.
        let val = self.0.borrow();
        if !visited.contains(self) {
            visited.insert(self.clone());
            val.operands.iter().for_each(|op| {
                let mut children = op.collect_operands_inner(visited);
                all_nodes.append(&mut children);
            });
            all_nodes.push(self.clone());
        }
        all_nodes
    }

    /// set the gradient value of all its
    /// children(all Values those which a calling Value is depending on) to 0.
    pub fn zero_grad(&self) {
        let mut visited = HashSet::new();
        self.zero_grad_inner(&mut visited);
    }

    /// set all the gradient values to zero
    pub fn zero_grad_inner(&self, visited: &mut HashSet<MVal>) {
        // if not already visited, then collect all of its children.
        let mut val = self.0.borrow_mut();
        val.grad = 0.0;
        if !visited.contains(self) {
            visited.insert(self.clone());
            val.operands.iter().for_each(|op| {
                op.zero_grad_inner(visited);
            });
        }
    }
}

/// Implement Hash based on the address of the Rc
impl Hash for MVal {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let ptr = Rc::as_ptr(&self.0) as *const (); // Get raw pointer address
        ptr.hash(state);
    }
}

/// Implement Eq and PartialEq based on the address of the Rc
impl PartialEq for MVal {
    fn eq(&self, other: &Self) -> bool {
        Rc::as_ptr(&self.0) == Rc::as_ptr(&other.0)
    }
}

impl Eq for MVal {}
