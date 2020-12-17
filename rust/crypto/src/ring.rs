use ndarray::prelude::*;
use rand::prelude::*;
use std::convert::TryInto;
use std::num::Wrapping;
use std::ops::{Add, Mul, Sub};

use crate::prng::{AesRng, PRNGSeed};

#[derive(Clone, Debug, PartialEq)]
pub struct Ring64Tensor(pub ArrayD<Wrapping<u64>>);

pub trait Sample {
    fn sample_uniform(shape: &[usize], key: &[u8]) -> Self;
}

impl Sample for Ring64Tensor {
    fn sample_uniform(shape: &[usize], key: &[u8]) -> Self {
        let seed: PRNGSeed = key.try_into().unwrap();
        let mut rng = AesRng::from_seed(seed);
        let length = shape.iter().product();
        let values: Vec<_> = (0..length).map(|_| Wrapping(rng.next_u64())).collect();
        let ix = IxDyn(shape);
        Ring64Tensor(Array::from_shape_vec(ix, values).unwrap())
    }
}

pub trait Fill {
    fn fill(shape: &[usize], el: u64) -> Self;
}

impl Fill for Ring64Tensor {
    fn fill(shape: &[usize], el: u64) -> Self {
        Ring64Tensor(ArrayD::from_elem(shape, Wrapping(el)))
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

pub struct Replicated<T>(T, T, T);

impl<T> Mul<Replicated<T>> for Replicated<T>
where
    T: Mul<T, Output = T>,
{
    type Output = Replicated<T>;
    fn mul(self, other: Replicated<T>) -> Self::Output {
        // TODO
        Replicated(self.0 * other.0, self.1 * other.1, self.2 * other.2)
    }
}

pub fn share(x: &Ring64Tensor) -> Replicated<Ring64Tensor> {
    // TODO
    Replicated(x.clone(), x.clone(), x.clone())
}

pub fn reconstruct<T>(x: Replicated<T>) -> T {
    // TODO
    x.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let a = Ring64Tensor::from(vec![1, 2, 3]);
        let b = Ring64Tensor::from(vec![4, 5, 6]);

        let a_shared = share(&a);
        let b_shared = share(&b);

        let c_shared = a_shared * b_shared;
        let c: Ring64Tensor = reconstruct(c_shared);
        assert_eq!(c, a * b);
    }

    #[test]
    fn ring_matrix_vector_prod() {
        let array_backing = array![[1, 2], [3, 4]]
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
        let x_backing = array![[1, 2], [3, 4]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let y_backing = array![[1, 0], [0, 1]]
            .into_dimensionality::<IxDyn>()
            .unwrap();
        let x = Ring64Tensor::from(x_backing);
        let y = Ring64Tensor::from(y_backing);
        let z = x.dot(y);

        let r_backing = array![[1, 2], [3, 4]]
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
    }

    #[test]
    fn ring_fill() {
        let r = Ring64Tensor::fill(&[2], 1);
        assert_eq!(r, Ring64Tensor::from(vec![1, 1]))
    }
}
