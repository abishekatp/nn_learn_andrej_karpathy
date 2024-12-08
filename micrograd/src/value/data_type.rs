pub type DataType = f64;
pub trait IntoValue {
    fn into_value(self) -> DataType;
}

// Blanket implementation for all types that can convert into DataType
impl<T> IntoValue for T
where
    T: Into<DataType>,
{
    fn into_value(self) -> DataType {
        self.into()
    }
}
