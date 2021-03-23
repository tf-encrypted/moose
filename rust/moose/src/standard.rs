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

impl<T> StandardTensor<T>
where
    T: LinalgScalar + Clone,
{
    pub fn dot(self, other: StandardTensor<T>) -> StandardTensor<T> {
        match self.0.ndim() {
            1 => match other.0.ndim() {
                1 => {
                    let l = self.0.into_dimensionality::<Ix1>().unwrap();
                    let r = other.0.into_dimensionality::<Ix1>().unwrap();
                    let res = Array::from_elem([], l.dot(&r))
                        .into_dimensionality::<IxDyn>()
                        .unwrap();
                    StandardTensor::<T>(res)
                }
                2 => {
                    let l = self.0.into_dimensionality::<Ix1>().unwrap();
                    let r = other.0.into_dimensionality::<Ix2>().unwrap();
                    let res = l.dot(&r).into_dimensionality::<IxDyn>().unwrap();
                    StandardTensor::<T>(res)
                }
                other_rank => panic!(
                    "Dot<StandardTensor> cannot handle argument of rank {:?} ",
                    other_rank
                ),
            },
            2 => match other.0.ndim() {
                1 => {
                    let l = self.0.into_dimensionality::<Ix2>().unwrap();
                    let r = other.0.into_dimensionality::<Ix1>().unwrap();
                    let res = l.dot(&r).into_dimensionality::<IxDyn>().unwrap();
                    StandardTensor::<T>(res)
                }
                2 => {
                    let l = self.0.into_dimensionality::<Ix2>().unwrap();
                    let r = other.0.into_dimensionality::<Ix2>().unwrap();
                    let res = l.dot(&r).into_dimensionality::<IxDyn>().unwrap();
                    StandardTensor::<T>(res)
                }
                other_rank => panic!(
                    "Dot<StandardTensor> cannot handle argument of rank {:?} ",
                    other_rank
                ),
            },
            other_rank => panic!(
                "Dot<StandardTensor> not implemented for tensors of rank {:?}",
                other_rank
            ),
        }
    }
}

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
    pub fn ones(shape: Shape) -> Self {
        StandardTensor::<T>(ArrayD::ones(shape.0))
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

impl<T> StandardTensor<T>
where
    T: LinalgScalar,
{
    pub fn sum(self, axis: Option<usize>) -> Self {
        if let Some(i) = axis {
            StandardTensor::<T>(self.0.sum_axis(Axis(i)))
        } else {
            let out = Array::from_elem([], self.0.sum())
                .into_dimensionality::<IxDyn>()
                .unwrap();
            StandardTensor::<T>(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_prod_f32() {
        let x = StandardTensor::<f32>::from(
            array![[1.0, -2.0], [3.0, -4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let y = x.clone();
        let z = x.dot(y);
        assert_eq!(
            z,
            StandardTensor::<f32>::from(
                array![[-5.0, 6.0], [-9.0, 10.0]]
                    .into_dimensionality::<IxDyn>()
                    .unwrap()
            )
        );
    }
}
