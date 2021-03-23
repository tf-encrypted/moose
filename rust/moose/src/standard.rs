// Standard dialect operations.

use crate::execution::Shape;
use ndarray::prelude::*;
use ndarray::LinalgScalar;
use serde::{Deserialize, Serialize};
use std::ops::{Add, Div, Mul, Sub};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct StandardTensor<T>(pub ArrayD<T>);

pub type Float32Tensor = StandardTensor<f32>;
pub type Float64Tensor = StandardTensor<f64>;
pub type Int8Tensor = StandardTensor<i8>;
pub type Int16Tensor = StandardTensor<i16>;
pub type Int32Tensor = StandardTensor<i32>;
pub type Int64Tensor = StandardTensor<i64>;
pub type Uint8Tensor = StandardTensor<u8>;
pub type Uint16Tensor = StandardTensor<u16>;
pub type Uint32Tensor = StandardTensor<u32>;
pub type Uint64Tensor = StandardTensor<u64>;

impl<T> From<ArrayD<T>> for StandardTensor<T>
where
    T: LinalgScalar,
{
    fn from(v: ArrayD<T>) -> StandardTensor<T> {
        StandardTensor::<T>(v)
    }
}

impl<T> Add for StandardTensor<T>
where
    T: LinalgScalar,
{
    type Output = StandardTensor<T>;
    fn add(self, other: StandardTensor<T>) -> Self::Output {
        StandardTensor::<T>(self.0 + other.0)
    }
}

impl<T> Sub for StandardTensor<T>
where
    T: LinalgScalar,
{
    type Output = StandardTensor<T>;
    fn sub(self, other: StandardTensor<T>) -> Self::Output {
        StandardTensor::<T>(self.0 - other.0)
    }
}

impl<T> Mul for StandardTensor<T>
where
    T: LinalgScalar,
{
    type Output = StandardTensor<T>;
    fn mul(self, other: StandardTensor<T>) -> Self::Output {
        StandardTensor::<T>(self.0 * other.0)
    }
}

impl<T> Div for StandardTensor<T>
where
    T: LinalgScalar,
{
    type Output = StandardTensor<T>;
    fn div(self, other: StandardTensor<T>) -> Self::Output {
        StandardTensor::<T>(self.0 / other.0)
    }
}

impl<T> StandardTensor<T>
where
    T: LinalgScalar,
{
    pub fn reshape(self, newshape: Shape) -> Self {
        StandardTensor::<T>(self.0.into_shape(newshape.0).unwrap()) // TODO need to be fix (unwrap)
    }
}
