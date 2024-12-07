use std::{
    cell::RefCell,
    fmt::{Debug, Display},
    ops::{Add, Div, Mul, Sub},
    rc::Rc,
};

/*
Note:
- Each operation implemented on the Value object will have its own gradient value.
- The gradient will tell you, how the small change in the input of the operator
will affect its output. This is called local gradient of the operator.
- We can construct a complex expression using multiple such operators,
then we will have local gradient for each operator and also the global gradient.
- global gradient is how the small change in the input of the particular operator will
affetch the output of the overall expression.

- This can be calculated using the chain rule of derivative. Suppose assume the following
example.

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

- Similarly we can derive the gradient of multiplication operator also.
*/

#[derive(Debug)]
enum Operator {
    None,
    Plus,
    Minus,
    Mul,
    Div,
    Tanh,
}

pub struct Value {
    data: f64,
    grad: f64, // grad(global gradient) field will have gradient of final output with respect to the Value(Self).
    operands: Vec<MutableValue>,
    operator: Operator,
}

// we need Rc, because we need to allow reuse of instance of MutableValue.
// we need RcCell, because we need to mutate the grad field in the Value for each MutableValue
/// Cloning of MutableValue is a cheap operation since it is acutally wrapper around Rc and RefCell
/// because of the same if you try to use the same Value in two places, then
/// it's gradient will be the accumulated sum of gradients of all the places it has been used.
#[derive(Clone)]
pub struct MutableValue(Rc<RefCell<Value>>);

impl Value {
    /// Cloning of returned value(MutableValue) is a cheap operation since it is acutally wrapper around Rc and RefCell
    pub fn new(data: f64) -> MutableValue {
        MutableValue(Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            operands: vec![],
            operator: Operator::None,
        })))
    }

    // calling compute gradient on a output value will comput the gradient value
    // w.r.t each one of its operands and store those gradients in the corresponding operands objects.
    fn comput_gradient(&mut self) {
        // println!("{:?}: {:?}", self.operator, self);

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
                        lhs.grad += out.grad * 1.0;
                        rhs.grad += out.grad * 1.0;
                    } else {
                        /*
                            since this is single threaded only other place we got the
                            mutable reference is just in the above line.

                            if y = x + x = 2x then dy/dx = 2;
                        */
                        lhs.grad += 2.0;
                    }
                }
            }
            Operator::Minus => {
                if out.operands.len() == 2 {
                    let mut lhs = out.operands[0].0.borrow_mut();
                    // handle: let c = a.clone() - a;
                    if let Ok(mut rhs) = out.operands[1].0.try_borrow_mut() {
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
                        denominator.grad += out.grad * -1.0 * numerator.data
                            / (denominator.data * denominator.data);
                    }
                    // else case: if y = x/x, then dy/dx = 0
                }
            }
            Operator::Tanh => {
                if out.operands.len() == 1 {
                    let mut input = out.operands[0].0.borrow_mut();

                    // y = tahh(x) then dy/dx = 1 - (tanh(x))^2
                    input.grad += 1.0 - (out.data * out.data);
                }
            }
            Operator::None => {}
        }
    }
}

impl MutableValue {
    pub fn backward(&mut self) {
        let mut val = self.0.borrow_mut();
        // set the gradient of output node as 1.(because making a small change in the output value will directly change the output value itself)
        val.grad = 1.0;
        // compute its children gradients
        val.comput_gradient();

        val.operands.iter_mut().for_each(|op| {
            op.backward_internal();
        });
    }

    fn backward_internal(&mut self) {
        let mut val = self.0.borrow_mut();
        val.comput_gradient();

        val.operands.iter_mut().for_each(|op| {
            op.backward_internal();
        });
    }
}

impl MutableValue {
    // here lifetime of Value would be same as the lifetime of the self.
    pub fn tanh(self) -> MutableValue {
        let val = self.0.borrow();
        MutableValue(Rc::new(RefCell::new(Value {
            data: val.data.tanh(),
            grad: 0.0,
            operands: vec![self.clone()],
            operator: Operator::Tanh,
        })))
    }
}

impl Add for MutableValue {
    type Output = MutableValue;

    fn add(self, rhs: Self) -> Self::Output {
        let lhsv = self.0.borrow();
        let rhsv = rhs.0.borrow();
        MutableValue(Rc::new(RefCell::new(Value {
            data: lhsv.data + rhsv.data,
            grad: 0.0,
            operands: vec![self.clone(), rhs.clone()],
            operator: Operator::Plus,
        })))
    }
}

impl Sub for MutableValue {
    type Output = MutableValue;

    fn sub(self, rhs: Self) -> Self::Output {
        let lhsv = self.0.borrow();
        let rhsv = rhs.0.borrow();
        MutableValue(Rc::new(RefCell::new(Value {
            data: lhsv.data - rhsv.data,
            grad: 0.0,
            operands: vec![self.clone(), rhs.clone()],
            operator: Operator::Minus,
        })))
    }
}

impl Mul for MutableValue {
    type Output = MutableValue;

    fn mul(self, rhs: Self) -> Self::Output {
        let lhsv = self.0.borrow();
        let rhsv = rhs.0.borrow();
        MutableValue(Rc::new(RefCell::new(Value {
            data: lhsv.data * rhsv.data,
            grad: 0.0,
            operands: vec![self.clone(), rhs.clone()],
            operator: Operator::Mul,
        })))
    }
}

impl Div for MutableValue {
    type Output = MutableValue;

    fn div(self, rhs: Self) -> Self::Output {
        let lhsv = self.0.borrow();
        let rhsv = rhs.0.borrow();
        MutableValue(Rc::new(RefCell::new(Value {
            data: lhsv.data / rhsv.data,
            grad: 0.0,
            operands: vec![self.clone(), rhs.clone()],
            operator: Operator::Div,
        })))
    }
}

impl Display for MutableValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value({})", self.0.borrow().data)
    }
}

impl Debug for MutableValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let val = self.0.borrow();
        write!(f, "Value(data:{}, grad:{})", val.data, val.grad)
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value({})", self.data)
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value(data:{}, grad:{})", self.data, self.grad)
    }
}
