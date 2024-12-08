use super::{
    data_type::{DataType, IntoValue},
    MutableValue, Operator, Value,
};
use std::{cell::RefCell, ops::Sub, rc::Rc};

// MutableValue - MutableValue
impl Sub for MutableValue {
    type Output = MutableValue;

    fn sub(self, rhs: Self) -> Self::Output {
        let lhsv = self.0.borrow().data;
        let rhsv = rhs.0.borrow().data;

        MutableValue(Rc::new(RefCell::new(Value {
            data: lhsv - rhsv,
            grad: 0.0,
            operands: vec![self.clone(), rhs.clone()],
            operator: Operator::Minus,
        })))
    }
}

// DataType - MutableValue
impl Sub<MutableValue> for DataType {
    type Output = MutableValue;

    fn sub(self, rhs: MutableValue) -> Self::Output {
        let lhsv = self;
        let lhs = MutableValue(Rc::new(RefCell::new(Value {
            data: lhsv,
            grad: 0.0,
            operands: vec![],
            operator: Operator::None,
        })));

        let rhsv = rhs.0.borrow().data;

        MutableValue(Rc::new(RefCell::new(Value {
            data: lhsv - rhsv,
            grad: 0.0,
            operands: vec![lhs, rhs.clone()],
            operator: Operator::Minus,
        })))
    }
}

// MutableValue - T
impl<T> Sub<T> for MutableValue
where
    T: IntoValue,
{
    type Output = MutableValue;

    fn sub(self, rhs: T) -> Self::Output {
        let lhs = self;
        let lhsv = lhs.0.borrow().data;

        let rhsv = rhs.into_value();
        let rhs = MutableValue(Rc::new(RefCell::new(Value {
            data: rhsv,
            grad: 0.0,
            operands: vec![],
            operator: Operator::None,
        })));

        MutableValue(Rc::new(RefCell::new(Value {
            data: lhsv - rhsv,
            grad: 0.0,
            operands: vec![lhs.clone(), rhs],
            operator: Operator::Minus,
        })))
    }
}
