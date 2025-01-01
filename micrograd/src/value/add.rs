use super::{
    data_type::{DataType, IntoValue},
    MutableValue, Operator, Value,
};
use std::{cell::RefCell, ops::Add, rc::Rc};

// MutableValue + MutableValue
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
            label: format!("({}+{})", lhsv.label, rhsv.label),
            visited: false,
        })))
    }
}

// lhs needs to be cloned if it needs to be reused after the operation.
// DataType + MutableValue
impl Add<MutableValue> for DataType {
    type Output = MutableValue;

    fn add(self, rhs: MutableValue) -> Self::Output {
        let lhsv = self;
        let lhs = MutableValue(Rc::new(RefCell::new(Value {
            data: lhsv,
            grad: 0.0,
            operands: vec![],
            operator: Operator::None,
            label: String::new(),
            visited: false,
        })));

        let rhsv = rhs.0.borrow();

        MutableValue(Rc::new(RefCell::new(Value {
            data: lhsv + rhsv.data,
            grad: 0.0,
            operands: vec![lhs, rhs.clone()],
            operator: Operator::Plus,
            label: format!("({}+{})", lhsv, rhsv.label),
            visited: false,
        })))
    }
}

// MutableValue + T
impl<T> Add<T> for MutableValue
where
    T: IntoValue,
{
    type Output = MutableValue;

    fn add(self, rhs: T) -> Self::Output {
        let lhs = self;
        let lhsv = lhs.0.borrow();

        let rhsv = rhs.into_value();
        let rhs = MutableValue(Rc::new(RefCell::new(Value {
            data: rhsv,
            grad: 0.0,
            operands: vec![],
            operator: Operator::None,
            label: String::new(),
            visited: false,
        })));

        MutableValue(Rc::new(RefCell::new(Value {
            data: lhsv.data + rhsv,
            grad: 0.0,
            operands: vec![lhs.clone(), rhs],
            operator: Operator::Plus,
            label: format!("({}+{})", lhsv.label, rhsv),
            visited: false,
        })))
    }
}
