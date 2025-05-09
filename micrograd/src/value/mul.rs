use super::{
    data_type::{DataType, IntoValue},
    MVal, Operator, Value,
};
use std::{cell::RefCell, ops::Mul, rc::Rc};

// MVal * MVal
impl Mul for MVal {
    type Output = MVal;

    fn mul(self, rhs: Self) -> Self::Output {
        let lhsv = self.0.borrow();
        let rhsv = rhs.0.borrow();

        MVal(Rc::new(RefCell::new(Value {
            data: lhsv.data * rhsv.data,
            grad: 0.0,
            operands: vec![self.clone(), rhs.clone()],
            operator: Operator::Mul,
            label: format!("({}*{})", lhsv.label, rhsv.label),
        })))
    }
}

// DataType * MVal
impl Mul<MVal> for DataType {
    type Output = MVal;

    fn mul(self, rhs: MVal) -> Self::Output {
        let lhsv = self;
        let lhs = MVal(Rc::new(RefCell::new(Value {
            data: lhsv,
            grad: 0.0,
            operands: vec![],
            operator: Operator::None,
            label: String::new(),
        })));

        let rhsv = rhs.0.borrow();

        MVal(Rc::new(RefCell::new(Value {
            data: lhsv * rhsv.data,
            grad: 0.0,
            operands: vec![lhs, rhs.clone()],
            operator: Operator::Mul,
            label: format!("({}*{})", lhsv, rhsv.label),
        })))
    }
}

// MVal * T
impl<T> Mul<T> for MVal
where
    T: IntoValue,
{
    type Output = MVal;

    fn mul(self, rhs: T) -> Self::Output {
        let lhs = self;
        let lhsv = lhs.0.borrow();

        let rhsv = rhs.into_value();
        let rhs = MVal(Rc::new(RefCell::new(Value {
            data: rhsv,
            grad: 0.0,
            operands: vec![],
            operator: Operator::None,
            label: String::new(),
        })));
        MVal(Rc::new(RefCell::new(Value {
            data: lhsv.data * rhsv,
            grad: 0.0,
            operands: vec![lhs.clone(), rhs],
            operator: Operator::Mul,
            label: format!("({}*{})", lhsv.label, rhsv),
        })))
    }
}
