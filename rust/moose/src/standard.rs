extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::prelude::*;
use ndarray::LinalgScalar;
use ndarray_linalg::types::{Lapack, Scalar};
use ndarray_linalg::*;
use num_traits::FromPrimitive;
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct Shape(pub Vec<usize>);

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

impl Shape {
    pub fn expand(mut self, axis: usize) -> Self {
        self.0.insert(axis, 1);
        self
    }

    pub fn slice(self, begin: usize, end: usize) -> Self {
        let slc = &self.0[begin..end];
        Shape(slc.to_vec())
    }
}

impl<T> StandardTensor<T>
where
    T: LinalgScalar,
{
    pub fn atleast_2d(self, to_column_vector: bool) -> StandardTensor<T> {
        match self.0.ndim() {
            0 => StandardTensor::<T>(self.0.into_shape(IxDyn(&[1, 1])).unwrap()),
            1 => {
                let length = self.0.len();
                let newshape = if to_column_vector {
                    IxDyn(&[length, 1])
                } else {
                    IxDyn(&[1, length])
                };
                StandardTensor::<T>(self.0.into_shape(newshape).unwrap())
            }
            2 => self,
            otherwise => panic!(
                "Tensor input for `atleast_2d` must have rank <= 2, found rank {:?}.",
                otherwise
            ),
        }
    }

    pub fn dot(self, other: StandardTensor<T>) -> StandardTensor<T> {
        match (self.0.ndim(), other.0.ndim()) {
            (1, 1) => {
                let l = self.0.into_dimensionality::<Ix1>().unwrap();
                let r = other.0.into_dimensionality::<Ix1>().unwrap();
                let res = Array::from_elem([], l.dot(&r))
                    .into_dimensionality::<IxDyn>()
                    .unwrap();
                StandardTensor::<T>(res)
            }
            (1, 2) => {
                let l = self.0.into_dimensionality::<Ix1>().unwrap();
                let r = other.0.into_dimensionality::<Ix2>().unwrap();
                let res = l.dot(&r).into_dimensionality::<IxDyn>().unwrap();
                StandardTensor::<T>(res)
            }
            (2, 1) => {
                let l = self.0.into_dimensionality::<Ix2>().unwrap();
                let r = other.0.into_dimensionality::<Ix1>().unwrap();
                let res = l.dot(&r).into_dimensionality::<IxDyn>().unwrap();
                StandardTensor::<T>(res)
            }
            (2, 2) => {
                let l = self.0.into_dimensionality::<Ix2>().unwrap();
                let r = other.0.into_dimensionality::<Ix2>().unwrap();
                let res = l.dot(&r).into_dimensionality::<IxDyn>().unwrap();
                StandardTensor::<T>(res)
            }
            (self_rank, other_rank) => panic!(
                // TODO: replace with proper error handling
                "Dot<StandardTensor> not implemented between tensors of rank {:?} and {:?}.",
                self_rank, other_rank,
            ),
        }
    }

    pub fn ones(shape: Shape) -> Self {
        StandardTensor::<T>(ArrayD::ones(shape.0))
    }

    pub fn reshape(self, newshape: Shape) -> Self {
        StandardTensor::<T>(self.0.into_shape(newshape.0).unwrap()) // TODO need to be fix (unwrap)
    }

    pub fn expand_dims(self, axis: usize) -> Self {
        let newshape = self.shape().expand(axis);
        self.reshape(newshape)
    }

    pub fn shape(&self) -> Shape {
        Shape(self.0.shape().into())
    }

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

    pub fn transpose(self) -> Self {
        StandardTensor::<T>(self.0.reversed_axes())
    }
}

impl<T> StandardTensor<T>
where
    T: LinalgScalar + FromPrimitive,
{
    pub fn mean(self, axis: Option<usize>) -> Self {
        match axis {
            Some(i) => {
                let reduced = self.0.mean_axis(Axis(i)).unwrap();
                StandardTensor::<T>(reduced)
            }
            None => {
                let mean = self.0.mean().unwrap();
                let out = Array::from_elem([], mean)
                    .into_dimensionality::<IxDyn>()
                    .unwrap();
                StandardTensor::<T>(out)
            }
        }
    }
}

impl<T> StandardTensor<T>
where
    T: Scalar + Lapack,
{
    pub fn inv(self) -> Self {
        match self.0.ndim() {
            2 => {
                let two_dim: Array2<T> = self.0.into_dimensionality::<Ix2>().unwrap();
                StandardTensor::<T>::from(
                    two_dim
                        .inv()
                        .unwrap()
                        .into_dimensionality::<IxDyn>()
                        .unwrap(),
                )
            }
            other_rank => panic!(
                "Inverse only defined for rank 2 matrices, not rank {:?}",
                other_rank,
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
        match self.0.broadcast(other.0.dim()) {
            Some(self_broadcasted) => StandardTensor::<T>(self_broadcasted.to_owned() / other.0),
            None => StandardTensor::<T>(self.0 / other.0),
        }
    }
}

impl<T> From<Vec<T>> for StandardTensor<T> {
    fn from(v: Vec<T>) -> StandardTensor<T> {
        StandardTensor(Array::from(v).into_dyn())
    }
}

impl<T> From<Array1<T>> for StandardTensor<T> {
    fn from(v: Array1<T>) -> StandardTensor<T> {
        StandardTensor(v.into_dyn())
    }
}

impl<T> From<Array2<T>> for StandardTensor<T> {
    fn from(v: Array2<T>) -> StandardTensor<T> {
        StandardTensor(v.into_dyn())
    }
}

pub fn concatenate<T>(axis: usize, arrays: &[StandardTensor<T>]) -> StandardTensor<T>
where
    T: LinalgScalar,
{
    let ax = Axis(axis);
    let inner_arrays: Vec<_> = arrays.iter().map(|a| a.0.view()).collect();

    let c = ndarray::concatenate(ax, &inner_arrays).unwrap();
    StandardTensor::<T>(c)
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

    #[test]
    fn test_inverse() {
        let x = StandardTensor::<f32>::from(
            array![[1.0, 2.0], [3.0, 4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );

        let x_inv = x.inv();

        assert_eq!(
            x_inv,
            StandardTensor::<f32>::from(
                array![[-2.0, 1.0], [1.5, -0.5]]
                    .into_dimensionality::<IxDyn>()
                    .unwrap()
            )
        );
    }

    #[test]
    fn test_shape_slice() {
        let x_shape = Shape(vec![1, 2, 3]);
        let x_slice = x_shape.slice(1, 3);
        assert_eq!(x_slice, Shape(vec![2, 3]))
    }

    #[test]
    fn test_transpose() {
        let x = StandardTensor::<f32>::from(
            array![[1.0, 2.0], [3.0, 4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let y = x.transpose();
        assert_eq!(
            y,
            StandardTensor::<f32>::from(
                array![[1.0, 3.0], [2.0, 4.0]]
                    .into_dimensionality::<IxDyn>()
                    .unwrap()
            )
        );
    }

    #[test]
    fn test_concatenate() {
        let a = StandardTensor::<f32>::from(
            array![[[1.0, 2.0], [3.0, 4.0]]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let b = StandardTensor::<f32>::from(
            array![[[1.0, 2.0], [3.0, 4.0]]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let expected = StandardTensor::<f32>::from(
            array![[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let conc = concatenate(0, &vec![a, b]);
        assert_eq!(conc, expected)
    }

    #[test]
    fn test_atleast_2d() {
        let a = StandardTensor::<f32>::from(
            array![[1.0, 2.0], [3.0, 4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let a_exp = a.clone();
        let b = StandardTensor::<f32>::from(
            array![1.0, 2.0, 3.0, 4.0]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let b_exp = StandardTensor::<f32>::from(
            array![[1.0, 2.0, 3.0, 4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let c = StandardTensor::<f32>::from(
            array![1.0, 2.0, 3.0, 4.0]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let c_exp = StandardTensor::<f32>::from(
            array![[1.0], [2.0], [3.0], [4.0]]
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let d = StandardTensor::<f32>::from(
            Array::from_elem([], 1.0)
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        );
        let d_exp =
            StandardTensor::<f32>::from(array![[1.0]].into_dimensionality::<IxDyn>().unwrap());
        let ax = a.atleast_2d(true);
        let bx = b.atleast_2d(false);
        let cx = c.atleast_2d(true);
        let dx = d.atleast_2d(true);
        assert_eq!(ax, a_exp);
        assert_eq!(bx, b_exp);
        assert_eq!(cx, c_exp);
        assert_eq!(dx, d_exp);
    }

    #[test]
    fn test_div() {
        let x_1 = StandardTensor::<f32>::from(array![1.0].into_dimensionality::<IxDyn>().unwrap());
        let y_1 =
            StandardTensor::<f32>::from(array![2.0, 4.0].into_dimensionality::<IxDyn>().unwrap());
        let z_1 = x_1.div(y_1);
        let z_1_exp =
            StandardTensor::<f32>::from(array![0.5, 0.25].into_dimensionality::<IxDyn>().unwrap());
        let x_2 =
            StandardTensor::<f32>::from(array![2.0, 4.0].into_dimensionality::<IxDyn>().unwrap());
        let y_2 = StandardTensor::<f32>::from(array![2.0].into_dimensionality::<IxDyn>().unwrap());
        let z_2 = x_2.div(y_2);
        let z_2_exp =
            StandardTensor::<f32>::from(array![1.0, 2.0].into_dimensionality::<IxDyn>().unwrap());

        assert_eq!(z_1, z_1_exp);
        assert_eq!(z_2, z_2_exp);
    }
}
