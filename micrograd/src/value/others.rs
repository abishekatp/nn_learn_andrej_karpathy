use super::{data_type::IntoValue, MVal, Operator, Value};
use std::{cell::RefCell, rc::Rc};

impl MVal {
    pub fn tanh(self) -> MVal {
        let val = self.0.borrow().data;
        MVal(Rc::new(RefCell::new(Value {
            data: val.tanh(),
            grad: 0.0,
            operands: vec![self.clone()],
            operator: Operator::Tanh,
            label: format!("tanh({})", val),
            visited: false,
        })))
    }

    pub fn exp(self) -> MVal {
        let val = self.0.borrow().data;
        MVal(Rc::new(RefCell::new(Value {
            data: val.exp(),
            grad: 0.0,
            operands: vec![self.clone()],
            operator: Operator::Exp,
            label: format!("exp({})", val),
            visited: false,
        })))
    }

    pub fn pow<T: IntoValue>(self, other: T) -> MVal {
        let val = self.0.borrow().data;

        let powv = other.into_value();
        let power = MVal(Rc::new(RefCell::new(Value {
            data: powv,
            grad: 0.0,
            operands: vec![],
            operator: Operator::None,
            label: String::new(),
            visited: false,
        })));

        MVal(Rc::new(RefCell::new(Value {
            data: val.powf(powv),
            grad: 0.0,
            operands: vec![self.clone(), power],
            operator: Operator::Pow,
            label: format!("pow({},{})", val, powv),
            visited: false,
        })))
    }
}
