use std::fmt::{Debug, Display};

use super::{MVal, Value};

impl Display for MVal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value({})", self.0.borrow().data)
    }
}

impl Debug for MVal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let val = self.0.borrow();
        write!(
            f,
            "Value(data:{}, grad:{}, {})",
            val.data, val.grad, val.label,
        )
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value({})", self.data)
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Value(data:{}, grad:{}, {})",
            self.data, self.grad, self.label,
        )
    }
}
