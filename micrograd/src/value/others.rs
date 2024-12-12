use super::{data_type::IntoValue, MutableValue, Operator, Value};
use std::{cell::RefCell, rc::Rc};

impl MutableValue {
    pub fn tanh(self) -> MutableValue {
        let val = self.0.borrow().data;
        MutableValue(Rc::new(RefCell::new(Value {
            data: val.tanh(),
            grad: 0.0,
            operands: vec![self.clone()],
            operator: Operator::Tanh,
            label: format!("tanh({})", val),
        })))
    }

    pub fn exp(self) -> MutableValue {
        let val = self.0.borrow().data;
        MutableValue(Rc::new(RefCell::new(Value {
            data: val.exp(),
            grad: 0.0,
            operands: vec![self.clone()],
            operator: Operator::Exp,
            label: format!("exp({})", val),
        })))
    }

    pub fn pow<T: IntoValue>(self, other: T) -> MutableValue {
        let val = self.0.borrow().data;

        let powv = other.into_value();
        let power = MutableValue(Rc::new(RefCell::new(Value {
            data: powv,
            grad: 0.0,
            operands: vec![],
            operator: Operator::None,
            label: String::new(),
        })));

        MutableValue(Rc::new(RefCell::new(Value {
            data: val.powf(powv),
            grad: 0.0,
            operands: vec![self.clone(), power],
            operator: Operator::Pow,
            label: format!("pow({},{})", val, powv),
        })))
    }
}
