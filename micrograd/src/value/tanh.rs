use super::{MutableValue, Operator, Value};
use std::{cell::RefCell, rc::Rc};

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
