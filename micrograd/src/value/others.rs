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
        })))
    }

    // Rectified Linear Unit y = max(0,x)
    pub fn relu(self) -> MVal {
        let val = self.0.borrow().data;
        MVal(Rc::new(RefCell::new(Value {
            data: if val < 0.0 { 0.0 } else { val },
            grad: 0.0,
            operands: vec![self.clone()],
            operator: Operator::ReLU,
            label: format!("ReLU({})", val),
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
        })));

        MVal(Rc::new(RefCell::new(Value {
            data: val.powf(powv),
            grad: 0.0,
            operands: vec![self.clone(), power],
            operator: Operator::Pow,
            label: format!("pow({},{})", val, powv),
        })))
    }
}
