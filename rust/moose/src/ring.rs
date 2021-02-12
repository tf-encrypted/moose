use ndarray::prelude::*;
use rand::prelude::*;
use std::convert::TryInto;
use std::num::Wrapping;
use std::ops::{Add, Mul, Shl, Shr, Sub};

use crate::prng::{AesRng, RngSeed};

use crate::bit::BitTensor;

#[derive(Clone, Debug, PartialEq)]
pub struct Ring64Tensor(pub ArrayD<Wrapping<u64>>);

pub trait Sample {
    fn sample_uniform(shape: &[usize], key: &[u8]) -> Self;
    fn sample_bits(shape: &[usize], key: &[u8]) -> Self;
}

impl Sample for Ring64Tensor {
    fn sample_uniform(shape: &[usize], key: &[u8]) -> Self {
        let seed: RngSeed = key.try_into().unwrap();
        let mut rng = AesRng::from_seed(seed);
        let length = shape.iter().product();
        let values: Vec<_> = (0..length).map(|_| Wrapping(rng.next_u64())).collect();
        let ix = IxDyn(shape);
        Ring64Tensor(Array::from_shape_vec(ix, values).unwrap())
    }
    fn sample_bits(shape: &[usize], key: &[u8]) -> Self {
        let seed: RngSeed = key.try_into().unwrap();
        let mut rng = AesRng::from_seed(seed);
        let length = shape.iter().product();
        let values: Vec<_> = (0..length)
            .map(|_| Wrapping(rng.get_bit() as u64))
            .collect();
        let ix = IxDyn(shape);
        Ring64Tensor(Array::from_shape_vec(ix, values).unwrap())
    }
}

impl Ring64Tensor {
    pub fn fill(shape: &[usize], el: u64) -> Ring64Tensor {
        Ring64Tensor(ArrayD::from_elem(shape, Wrapping(el)))
    }
}

pub trait BitInjector {
    fn bit_inject(x: BitTensor, bit_idx: usize) -> Self;
}

impl BitInjector for Ring64Tensor {
    fn bit_inject(x: BitTensor, bit_idx: usize) -> Self {
        let ring_rep = x.0.mapv(|ai| ((ai as u64) << bit_idx));
        Ring64Tensor::from(ring_rep)
    }
}

impl From<ArrayD<i64>> for Ring64Tensor {
    fn from(a: ArrayD<i64>) -> Ring64Tensor {
        let wrapped = a.mapv(|ai| Wrapping(ai as u64));
        Ring64Tensor(wrapped)
    }
}

impl From<&Ring64Tensor> for ArrayD<i64> {
    fn from(r: &Ring64Tensor) -> ArrayD<i64> {
        r.0.mapv(|element| element.0 as i64)
    }
}

impl From<ArrayD<u64>> for Ring64Tensor {
    fn from(a: ArrayD<u64>) -> Ring64Tensor {
        let wrapped = a.mapv(Wrapping);
        Ring64Tensor(wrapped)
    }
}

impl From<Vec<u64>> for Ring64Tensor {
    fn from(v: Vec<u64>) -> Ring64Tensor {
        let ix = IxDyn(&[v.len()]);
        use vec_utils::VecExt;
        let v_wrapped: Vec<_> = v.map(Wrapping);
        Ring64Tensor(Array::from_shape_vec(ix, v_wrapped).unwrap())
    }
}

impl From<&[u64]> for Ring64Tensor {
    fn from(v: &[u64]) -> Ring64Tensor {
        let ix = IxDyn(&[v.len()]);
        let v_wrapped: Vec<_> = v.iter().map(|vi| Wrapping(*vi)).collect();
        Ring64Tensor(Array::from_shape_vec(ix, v_wrapped).unwrap())
    }
}

impl Add for Ring64Tensor {
    type Output = Ring64Tensor;
    fn add(self, other: Self) -> Self::Output {
        Ring64Tensor(self.0.add(other.0))
    }
}

impl Mul for Ring64Tensor {
    type Output = Ring64Tensor;
    fn mul(self, other: Self) -> Self::Output {
        Ring64Tensor(self.0.mul(other.0))
    }
}

impl Sub for Ring64Tensor {
    type Output = Ring64Tensor;
    fn sub(self, other: Self) -> Self::Output {
        Ring64Tensor(self.0.sub(other.0))
    }
}

impl Shl<usize> for Ring64Tensor {
    type Output = Ring64Tensor;
    fn shl(self, other: usize) -> Self::Output {
        Ring64Tensor(self.0 << other)
    }
}

impl Shr<usize> for Ring64Tensor {
    type Output = Ring64Tensor;
    fn shr(self, other: usize) -> Self::Output {
        Ring64Tensor(self.0 >> other)
    }
}

pub trait Dot<Rhs> {
    type Output;
    fn dot(self, rhs: Rhs) -> Self::Output;
}

impl Dot<Ring64Tensor> for Ring64Tensor {
    type Output = Ring64Tensor;
    fn dot(self, rhs: Ring64Tensor) -> Self::Output {
        match self.0.ndim() {
            1 => match rhs.0.ndim() {
                1 => {
                    let l = self.0.into_dimensionality::<Ix1>().unwrap();
                    let r = rhs.0.into_dimensionality::<Ix1>().unwrap();
                    let res = Array::from_elem([], l.dot(&r))
                        .into_dimensionality::<IxDyn>()
                        .unwrap();
                    Ring64Tensor(res)
                }
                2 => {
                    let l = self.0.into_dimensionality::<Ix1>().unwrap();
                    let r = rhs.0.into_dimensionality::<Ix2>().unwrap();
                    let res = l.dot(&r).into_dimensionality::<IxDyn>().unwrap();
                    Ring64Tensor(res)
                }
                other => panic!(
                    "Dot<Ring64Tensor> cannot handle argument of rank {:?} ",
                    other
                ),
            },
            2 => match rhs.0.ndim() {
                1 => {
                    let l = self.0.into_dimensionality::<Ix2>().unwrap();
                    let r = rhs.0.into_dimensionality::<Ix1>().unwrap();
                    let res = l.dot(&r).into_dimensionality::<IxDyn>().unwrap();
                    Ring64Tensor(res)
                }
                2 => {
                    let l = self.0.into_dimensionality::<Ix2>().unwrap();
                    let r = rhs.0.into_dimensionality::<Ix2>().unwrap();
                    let res = l.dot(&r).into_dimensionality::<IxDyn>().unwrap();
                    Ring64Tensor(res)
                }
                other => panic!(
                    "Dot<Ring64Tensor> cannot handle argument of rank {:?} ",
                    other
                ),
            },
            other => panic!(
                "Dot<Ring64Tensor> not implemented for tensors of rank {:?}",
                other
            ),
        }
    }
}

impl Ring64Tensor {
    pub fn sum(self, axis: Option<usize>) -> Ring64Tensor {
        if let Some(i) = axis {
            Ring64Tensor(self.0.sum_axis(Axis(i)))
        } else {
            let out = Array::from_elem([], self.0.sum())
                .into_dimensionality::<IxDyn>()
                .unwrap();
            Ring64Tensor(out)
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
        let result = Ring64Tensor(r_backing);
        assert_eq!(result, z)
    }

    #[test]
    fn ring_sample() {
        let key = [0u8; 16];
        let r = Ring64Tensor::sample_uniform(&[5], &key);
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

        let r_bits = Ring64Tensor::sample_bits(&[5], &key);
        assert_eq!(r_bits, Ring64Tensor::from(vec![0, 1, 1, 0, 0]));
    }

    #[test]
    fn ring_fill() {
        let r = Ring64Tensor::fill(&[2], 1);
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
}
