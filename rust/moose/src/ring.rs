use ndarray::prelude::*;
use ndarray::LinalgScalar;
use num_traits::Zero;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::num::Wrapping;
use std::ops::{Add, Mul, Shl, Shr, Sub};

use crate::bit::BitTensor;
use crate::computation::HostPlacement;
use crate::computation::Role;
use crate::prim::RawSeed;
use crate::prng::AesRng;
use crate::standard::{RawShape, Shape};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct AbstractRingTensor<T>(pub ArrayD<Wrapping<T>>, HostPlacement);

pub type Ring64Tensor = AbstractRingTensor<u64>;

pub type Ring128Tensor = AbstractRingTensor<u128>;

impl Ring64Tensor {
    pub fn sample_uniform(shape: &RawShape, seed: &RawSeed) -> Ring64Tensor {
        let mut rng = AesRng::from_seed(seed.0);
        let size = shape.0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.next_u64())).collect();
        let ix = IxDyn(shape.0.as_ref());
        Ring64Tensor::new(Array::from_shape_vec(ix, values).unwrap())
    }
    pub fn sample_bits(shape: &RawShape, seed: &RawSeed) -> Self {
        let mut rng = AesRng::from_seed(seed.0);
        let size = shape.0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.get_bit() as u64)).collect();
        let ix = IxDyn(shape.0.as_ref());
        Ring64Tensor::new(Array::from_shape_vec(ix, values).unwrap())
    }
}

impl Ring128Tensor {
    pub fn sample_uniform(shape: &RawShape, seed: &RawSeed) -> Self {
        let mut rng = AesRng::from_seed(seed.0);
        let size = shape.0.iter().product();
        let values: Vec<_> = (0..size)
            .map(|_| Wrapping(((rng.next_u64() as u128) << 64) + rng.next_u64() as u128))
            .collect();
        let ix = IxDyn(shape.0.as_ref());
        Ring128Tensor::new(Array::from_shape_vec(ix, values).unwrap())
    }

    pub fn sample_bits(shape: &RawShape, seed: &RawSeed) -> Self {
        let mut rng = AesRng::from_seed(seed.0);
        let size = shape.0.iter().product();
        let values: Vec<_> = (0..size).map(|_| Wrapping(rng.get_bit() as u128)).collect();
        let ix = IxDyn(shape.0.as_ref());
        Ring128Tensor::new(Array::from_shape_vec(ix, values).unwrap())
    }
}

impl AbstractRingTensor<u64> {
    pub fn bit_extract(&self, bit_idx: usize) -> BitTensor {
        let temp = &self.0 >> bit_idx;
        let lsb = temp.mapv(|ai| (ai.0 & 1) as u8);
        BitTensor::from(lsb)
    }
}

impl AbstractRingTensor<u128> {
    pub fn bit_extract(&self, bit_idx: usize) -> BitTensor {
        let temp = &self.0 >> bit_idx;
        let lsb = temp.mapv(|ai| (ai.0 & 1) as u8);
        BitTensor::from(lsb)
    }
}

impl<T> AbstractRingTensor<T>
where
    Wrapping<T>: Clone,
{
    pub fn fill(shape: &RawShape, el: T) -> AbstractRingTensor<T> {
        AbstractRingTensor(
            ArrayD::from_elem(shape.0.as_ref(), Wrapping(el)),
            HostPlacement {
                owner: Role::from("TODO"),
            },
        )
    }
}

impl<T> AbstractRingTensor<T> {
    pub fn shape(&self) -> Shape {
        Shape(RawShape(self.0.shape().into()), self.1.clone().into())
    }
}

impl<T> From<ArrayD<T>> for AbstractRingTensor<T>
where
    T: Clone,
{
    fn from(a: ArrayD<T>) -> AbstractRingTensor<T> {
        let wrapped = a.mapv(Wrapping);
        AbstractRingTensor(
            wrapped,
            HostPlacement {
                owner: Role::from("TODO"),
            },
        )
    }
}

impl From<ArrayD<i64>> for AbstractRingTensor<u64> {
    fn from(a: ArrayD<i64>) -> AbstractRingTensor<u64> {
        let ring_rep = a.mapv(|ai| Wrapping(ai as u64));
        AbstractRingTensor(
            ring_rep,
            HostPlacement {
                owner: Role::from("TODO"),
            },
        )
    }
}

impl From<ArrayD<i128>> for AbstractRingTensor<u128> {
    fn from(a: ArrayD<i128>) -> AbstractRingTensor<u128> {
        let ring_rep = a.mapv(|ai| Wrapping(ai as u128));
        AbstractRingTensor(
            ring_rep,
            HostPlacement {
                owner: Role::from("TODO"),
            },
        )
    }
}

impl<T> AbstractRingTensor<T> {
    pub fn new(a: ArrayD<Wrapping<T>>) -> AbstractRingTensor<T> {
        AbstractRingTensor(
            a,
            HostPlacement {
                owner: Role::from("TODO"),
            },
        )
    }
}

impl<T> From<BitTensor> for AbstractRingTensor<T>
where
    T: From<u8>,
{
    fn from(b: BitTensor) -> AbstractRingTensor<T> {
        let ring_rep = b.0.mapv(|ai| Wrapping(ai.into()));
        AbstractRingTensor(
            ring_rep,
            HostPlacement {
                owner: Role::from("TODO"),
            },
        )
    }
}

impl From<&AbstractRingTensor<u64>> for ArrayD<i64> {
    fn from(r: &AbstractRingTensor<u64>) -> ArrayD<i64> {
        r.0.mapv(|element| element.0 as i64)
    }
}

impl From<&AbstractRingTensor<u128>> for ArrayD<i128> {
    fn from(r: &AbstractRingTensor<u128>) -> ArrayD<i128> {
        r.0.mapv(|element| element.0 as i128)
    }
}

impl<T> From<Vec<T>> for AbstractRingTensor<T> {
    fn from(v: Vec<T>) -> AbstractRingTensor<T> {
        let ix = IxDyn(&[v.len()]);
        use vec_utils::VecExt;
        let v_wrapped: Vec<_> = v.map(Wrapping);
        AbstractRingTensor(
            Array::from_shape_vec(ix, v_wrapped).unwrap(),
            HostPlacement {
                owner: Role::from("TODO"),
            },
        )
    }
}

impl<T> From<&[T]> for AbstractRingTensor<T>
where
    T: Copy,
{
    fn from(v: &[T]) -> AbstractRingTensor<T> {
        let ix = IxDyn(&[v.len()]);
        let v_wrapped: Vec<_> = v.iter().map(|vi| Wrapping(*vi)).collect();
        AbstractRingTensor(
            Array::from_shape_vec(ix, v_wrapped).unwrap(),
            HostPlacement {
                owner: Role::from("TODO"),
            },
        )
    }
}

impl<T> Add<AbstractRingTensor<T>> for AbstractRingTensor<T>
where
    Wrapping<T>: Clone,
    Wrapping<T>: Add<Wrapping<T>, Output = Wrapping<T>>,
{
    type Output = AbstractRingTensor<T>;
    fn add(self, other: AbstractRingTensor<T>) -> Self::Output {
        AbstractRingTensor(self.0 + other.0, self.1)
    }
}

impl<T> Mul<AbstractRingTensor<T>> for AbstractRingTensor<T>
where
    Wrapping<T>: Clone,
    Wrapping<T>: Mul<Wrapping<T>, Output = Wrapping<T>>,
{
    type Output = AbstractRingTensor<T>;
    fn mul(self, other: AbstractRingTensor<T>) -> Self::Output {
        AbstractRingTensor(self.0 * other.0, self.1)
    }
}

impl<T> Sub<AbstractRingTensor<T>> for AbstractRingTensor<T>
where
    Wrapping<T>: Clone,
    Wrapping<T>: Sub<Wrapping<T>, Output = Wrapping<T>>,
{
    type Output = AbstractRingTensor<T>;
    fn sub(self, other: AbstractRingTensor<T>) -> Self::Output {
        AbstractRingTensor(self.0 - other.0, self.1)
    }
}

impl<T> Shl<usize> for AbstractRingTensor<T>
where
    Wrapping<T>: Clone,
    Wrapping<T>: Shl<usize, Output = Wrapping<T>>,
{
    type Output = AbstractRingTensor<T>;
    fn shl(self, other: usize) -> Self::Output {
        AbstractRingTensor(self.0 << other, self.1)
    }
}

impl<T> Shr<usize> for AbstractRingTensor<T>
where
    Wrapping<T>: Clone,
    Wrapping<T>: Shr<usize, Output = Wrapping<T>>,
{
    type Output = AbstractRingTensor<T>;
    fn shr(self, other: usize) -> Self::Output {
        AbstractRingTensor(self.0 >> other, self.1)
    }
}

impl<T> AbstractRingTensor<T>
where
    Wrapping<T>: LinalgScalar,
{
    pub fn dot(self, rhs: AbstractRingTensor<T>) -> AbstractRingTensor<T> {
        match self.0.ndim() {
            1 => match rhs.0.ndim() {
                1 => {
                    let l = self.0.into_dimensionality::<Ix1>().unwrap();
                    let r = rhs.0.into_dimensionality::<Ix1>().unwrap();
                    let res = Array::from_elem([], l.dot(&r))
                        .into_dimensionality::<IxDyn>()
                        .unwrap();
                    AbstractRingTensor(res, self.1)
                }
                2 => {
                    let l = self.0.into_dimensionality::<Ix1>().unwrap();
                    let r = rhs.0.into_dimensionality::<Ix2>().unwrap();
                    let res = l.dot(&r).into_dimensionality::<IxDyn>().unwrap();
                    AbstractRingTensor(res, self.1)
                }
                other => panic!(
                    "Dot<AbstractRingTensor> cannot handle argument of rank {:?} ",
                    other
                ),
            },
            2 => match rhs.0.ndim() {
                1 => {
                    let l = self.0.into_dimensionality::<Ix2>().unwrap();
                    let r = rhs.0.into_dimensionality::<Ix1>().unwrap();
                    let res = l.dot(&r).into_dimensionality::<IxDyn>().unwrap();
                    AbstractRingTensor(res, self.1)
                }
                2 => {
                    let l = self.0.into_dimensionality::<Ix2>().unwrap();
                    let r = rhs.0.into_dimensionality::<Ix2>().unwrap();
                    let res = l.dot(&r).into_dimensionality::<IxDyn>().unwrap();
                    AbstractRingTensor(res, self.1)
                }
                other => panic!(
                    "Dot<AbstractRingTensor> cannot handle argument of rank {:?} ",
                    other
                ),
            },
            other => panic!(
                "Dot<AbstractRingTensor> not implemented for tensors of rank {:?}",
                other
            ),
        }
    }
}

impl<T> AbstractRingTensor<T>
where
    Wrapping<T>: Clone + Zero,
{
    pub fn sum(self, axis: Option<usize>) -> AbstractRingTensor<T> {
        if let Some(i) = axis {
            AbstractRingTensor(self.0.sum_axis(Axis(i)), self.1)
        } else {
            let out = Array::from_elem([], self.0.sum())
                .into_dimensionality::<IxDyn>()
                .unwrap();
            AbstractRingTensor(out, self.1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_matrix_vector_prod() {
        let array_backing: ArrayD<i64> = array![[1, 2], [3, 4]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let x = Ring64Tensor::from(array_backing);
        let y = Ring64Tensor::from(vec![1, 1]);
        let z = x.dot(y);

        let result = Ring64Tensor::from(vec![3, 7]);
        assert_eq!(result, z)
    }

    #[test]
    fn ring_matrix_matrix_prod() {
        let x_backing: ArrayD<i64> = array![[1, 2], [3, 4]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let y_backing: ArrayD<i64> = array![[1, 0], [0, 1]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let x = Ring64Tensor::from(x_backing);
        let y = Ring64Tensor::from(y_backing);
        let z = x.dot(y);

        let r_backing: ArrayD<i64> = array![[1, 2], [3, 4]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let result = Ring64Tensor::from(r_backing);
        assert_eq!(result, z)
    }

    #[test]
    fn ring_vector_prod() {
        let x_backing = vec![1, 2];
        let y_backing = vec![1, 1];
        let x = Ring64Tensor::from(x_backing);
        let y = Ring64Tensor::from(y_backing);
        let z = x.dot(y);

        let r_backing = Array::from_elem([], Wrapping(3))
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let result = Ring64Tensor::new(r_backing);
        assert_eq!(result, z)
    }

    #[test]
    fn ring_sample() {
        let shape = RawShape(vec![5]);
        let seed = RawSeed([0u8; 16]);
        let r = Ring64Tensor::sample_uniform(&shape, &seed);
        assert_eq!(
            r,
            Ring64Tensor::from(vec![
                4263935709876578662,
                3326810793440857224,
                17325099178452873543,
                15208531650305571673,
                9619880027406922172
            ])
        );

        let r128 = Ring128Tensor::sample_uniform(&shape, &seed);
        assert_eq!(
            r128,
            Ring128Tensor::from(vec![
                78655730786844307471556614669614075016,
                319591670596555766473793801091584867161,
                177455464885365520564027128957528354027,
                72628979995024532377123578937486303732,
                299726520301515014350190124791858941972
            ])
        );

        let r_bits = Ring64Tensor::sample_bits(&shape, &seed);
        assert_eq!(r_bits, Ring64Tensor::from(vec![0, 1, 1, 0, 0]));

        let r128_bits = Ring128Tensor::sample_bits(&shape, &seed);
        assert_eq!(r128_bits, Ring128Tensor::from(vec![0, 1, 1, 0, 0]));
    }

    #[test]
    fn ring_fill() {
        let r = Ring64Tensor::fill(&RawShape(vec![2]), 1);
        assert_eq!(r, Ring64Tensor::from(vec![1, 1]))
    }

    #[test]
    fn ring_sum_with_axis() {
        let x_backing: ArrayD<i64> = array![[1, 2], [3, 4]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let x = Ring64Tensor::from(x_backing);
        let out = x.sum(Some(0));
        assert_eq!(out, Ring64Tensor::from(vec![4, 6]))
    }

    #[test]
    fn ring_sum_without_axis() {
        let x_backing: ArrayD<i64> = array![[1, 2], [3, 4]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let x = Ring64Tensor::from(x_backing);
        let exp_v: u64 = 10;
        let exp_backing = Array::from_elem([], exp_v)
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let exp = Ring64Tensor::from(exp_backing);
        let out = x.sum(None);
        assert_eq!(out, exp)
    }

    #[test]
    fn bit_extract() {
        let shape = RawShape(vec![5]);
        let value = 7;

        let r0 = Ring64Tensor::fill(&shape, value).bit_extract(0);
        assert_eq!(BitTensor::fill(&shape, 1), r0,);

        let r1 = Ring64Tensor::fill(&shape, value).bit_extract(1);
        assert_eq!(BitTensor::fill(&shape, 1), r1,);

        let r2 = Ring64Tensor::fill(&shape, value).bit_extract(2);
        assert_eq!(BitTensor::fill(&shape, 1), r2,);

        let r3 = Ring64Tensor::fill(&shape, value).bit_extract(3);
        assert_eq!(BitTensor::fill(&shape, 0), r3,)
    }
}
